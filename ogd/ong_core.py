from ogd.tools import *
import pytorch_lightning as pl
from models.mlp import MLP_ONG

alpha = 0.00005
class AgentModel_ONG_MINE(pl.LightningModule):
    def __init__(self, config, wandb_run, val_loaders):
        print("🥶🥶🥶🥶 Using my AgentModel")
        super().__init__()
        self.m = config.batch_size
        self.in_dim = 1024 ## YY: hardcoding
        self.criterion_fn = nn.CrossEntropyLoss()
        self.task_id = 0 ## YY: initialized task_id

        self.config = config
        self.wandb_run = wandb_run

        self.model = self.create_model()
        print(f"### The model is {self.model} ###")
        print(f"### The model has {count_parameter(self.model)} parameters ###")

        n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        # self.ogd_basis = None
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]))

        if self.config.gpu:
            # self.ogd_basis = self.ogd_basis.cuda()
            self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())

        self.val_loaders = val_loaders

        # Store initial Neural Tangents

        self.task_count = 0
        self.task_memory = {}
        self.task_mem_cache = {}

        self.task_grad_memory = {}
        self.task_grad_mem_cache = {}

        self.mem_loaders = list()

        self.A = []
        self.G = []
        self.U_A = []
        self.U_G = []
        for Wi in self.model.W:
            self.A.append(torch.zeros(Wi.size(1)))
            self.G.append(torch.zeros(Wi.size(0)))
        self.eps = 1e-1
        self.eigbasis_comp_freq = 50

    def create_model(self):
        cfg = self.config
        model  = MLP_ONG(hidden_dim=self.config.hidden_dim)
        if self.config.gpu :
            device = torch.device("cuda")
            model.to(device)
        print(f"🥶🥶🥶🥶 The model is {model}")
        return model

    def _get_new_ogd_basis(self, train_loader, last=False):
        return self._get_neural_tangents(train_loader,
                                         gpu=self.config.gpu,
                                         optimizer=self.optimizer,
                                         model=self.model, last=last)

    # YY: we iterate over a loader on the current task, collect some grad_vecs, and add this to new_basis_tensor
    # later on in the code, this is appropriately combined with previous task's grad_vecs and orthonormalized
    # however, ensuring its "naturalized" here should ensure our task gradient memory is always valid
    def _get_neural_tangents(self, train_loader, gpu, optimizer, model, last):
        new_basis = []

        for i, (inputs, targets, tasks) in tqdm(enumerate(train_loader),
                                                desc="get neural tangents",
                                                total=len(train_loader.dataset)):
            # if gpu:
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)

            inputs = inputs.view(-1, self.in_dim)
            out = self.forward(x=inputs, task=(tasks))
            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            # grad_vec = parameters_to_grad_vector(self.get_params_dict(last=last))
            grad_vec = self.get_natural_gradient_vector()
            new_basis.append(grad_vec)
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    def forward(self, x, task):
        task_key = task[0]
        out = self.model.forward(x)
        z, cache = out
        return z
        # if self.config.is_split :
        #     return out[task_key]
        # else :
        #     return out["All"]
    
    def forward_with_cache(self, x, task):
        task_key = task[0]
        out = self.model.forward(x)
        z, cache = out
        return z, cache

    def predict(self, x, task):
        x = torch.tensor(x)
        x = self.to_device(x)
        x = x.view(-1, self.in_dim)
        out = self.forward(x, task)
        z, cache = out
        _, pred = z.topk(1, 1, True, True)
        return pred

    def training_step(self, batch, batch_nb):
        self.model.train()
        assert self.model.training

        data, target, task = batch
        data = data.view(-1, self.in_dim)
        z, cache = self.forward_with_cache(data, task)
        a1, h1, a2, h2 = cache
        loss = self.criterion_fn(z, target)

        self.task_id = int(task[0])
        if (batch_nb) % self.eigbasis_comp_freq == 0:
            ## YY: need to recompute the gradients since lightning is clearing the gradients right before training_step is called...
            g_a1, g_a2, g_z = torch.autograd.grad(
                outputs=loss,
                inputs=(a1, a2, z),
                retain_graph=True,  # if you need to reuse the graph later
                create_graph=False
                )
            
            device = g_a1.device
            data = data.to(device)
            h1   = h1.to(device)
            h2   = h2.to(device)

            G1_ = 1/self.m * g_a1.t() @ g_a1
            A1_ = 1/self.m * data.t() @ data
            G2_ = 1/self.m * g_a2.t() @ g_a2
            A2_ = 1/self.m * h1.t() @ h1
            G3_ = 1/self.m * g_z.t() @ g_z
            A3_ = 1/self.m * h2.t() @ h2

            A_ = [A1_, A2_, A3_]
            G_ = [G1_, G2_, G3_]

            # Update running estimates of KFAC
            rho = min(1-1/(batch_nb+1), 0.95)

            for k in range(3):
                self.A[k] = rho*self.A[k].to(device) + (1-rho)*A_[k]
                self.G[k] = rho*self.G[k].to(device) + (1-rho)*G_[k]
        
            self.U_A = []
            self.U_G = []
            for k in range(3):                
                damp = self.eps  # or a slightly larger 1e-2 … 1e-1
                self.G[k] += damp * torch.eye(self.G[k].size(0), device=self.G[k].device)
                self.A[k] += damp * torch.eye(self.A[k].size(0), device=self.A[k].device)
                
                _, U_Ak = torch.linalg.eigh(self.A[k])
                _, U_Gk = torch.linalg.eigh(self.G[k])
                self.U_A.append(U_Ak)
                self.U_G.append(U_Gk)


        if batch_nb % self.config.val_check_interval == 0 and not self.config.no_val:
            log_dict = dict()

            # print(f"keys {self.val_loaders.keys()}, {self.mem_loaders.keys()}")

            for task_id_val in range(1, self.task_id + 1):
                val_acc = validate(self.val_loaders[task_id_val - 1],
                                   model=self,
                                   gpu=self.config.gpu,
                                   size=self.config.val_size)
                log_dict[f"val_acc_{task_id_val}"] = val_acc

            if self.config.ogd or self.config.ogd_plus :
                for task_id_val in range(1, self.task_id):
                    # TODO : Add mem val acc
                    val_acc = validate(self.mem_loaders[task_id_val - 1],
                                       model=self,
                                       gpu=self.config.gpu,
                                       size=self.config.val_size)
                    log_dict[f"mem_val_acc_{task_id_val}"] = val_acc


            # print(log_dict)
            log_dict["task_id"] = self.task_id
            wandb.log(log_dict)

        wandb.log({f"train_loss_{self.task_id}": loss,
                   "task_id": self.task_id})
        return {'loss': loss}

    def configure_optimizers(self):
        # Unused :)
        n_trainable = get_n_trainable(self.model)
        print(f"The model has {n_trainable} trainable parameters")
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.config.lr,
                                             momentum=0,
                                             weight_decay=0)
        return self.optimizer

    def get_params_dict(self, last, task_key=None):
        if self.config.is_split_cub :
            if last :
                return self.model.last[task_key].parameters()
            else :
                return self.model.linear.parameters()
        elif self.config.is_split :
            if last:
                return self.model.last[task_key].parameters()
            else:
                return self.model.linear.parameters()
        else:
            return self.model.parameters()

    def to_device(self, tensor):
        if self.config.gpu :
            return tensor.cuda()
        else :
            return tensor
        

    def inner_product(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute u^T F v under the EKFAC approximation F ≈ ⊕_k (A_k ⊗ G_k).

        Assumes:
        - self.model.W has 3 weight matrices [W1,W2,W3].
        - self.A[k], self.G[k] are the current (in_dim×in_dim)/(out_dim×out_dim) covariances.
        - self.U_A[k], self.U_G[k] are their eigenvector matrices.
        - u and v are 1D vectors of length sum_k (out_dim_k*in_dim_k),
            in the same flatten-by-layer order as you use everywhere else.
        """
        total = torch.tensor(0.0, device=u.device)
        offset = 0
        for k, W in enumerate(self.model.W):
            out_dim, in_dim = W.shape
            block_size = out_dim * in_dim

            # slice out this layer’s block and reshape
            u_k = u[offset:offset+block_size].view(out_dim, in_dim)
            v_k = v[offset:offset+block_size].view(out_dim, in_dim)
            offset += block_size

            # grab the EKFAC factors for this layer
            Gk = self.G[k]           # (out_dim, out_dim)
            Ak = self.A[k]           # (in_dim,  in_dim)
            UG = self.U_G[k]         # eigenvectors of Gk
            UA = self.U_A[k]         # eigenvectors of Ak

            # project u_k, v_k into the Kronecker‐eigenbasis
            #  u_tilde[i,j] = (UG^T u_k UA)[i,j]
            u_tilde = UG.t().matmul(u_k).matmul(UA)
            v_tilde = UG.t().matmul(v_k).matmul(UA)

            # get the eigenvalues of Gk and Ak:
            #  lam_G[i] = (UG^T Gk UG)[i,i], similarly lam_A[j]
            lam_G = torch.diagonal(UG.t().matmul(Gk).matmul(UG))
            lam_A = torch.diagonal(UA.t().matmul(Ak).matmul(UA))

            # now F’s eigenvalues on the (i,j) Kronecker‐basis are lam_G[i] * lam_A[j],
            # so the partial inner product is
            #   sum_{i,j} [lam_G[i]*lam_A[j] * u_tilde[i,j] * v_tilde[i,j]]
            total += (lam_G.unsqueeze(1) * lam_A.unsqueeze(0) * u_tilde * v_tilde).sum()
        return total
    
    # YY: upon time of calling, goes through all of self.model's layers and uses the EKFAC method to compute the natural gradient,
    # final return is one big gradient vector
    def get_natural_gradient_vector(self):
        vec = []
        for k in range(3):
            # bring the eigenbases and gradient onto GPU
            U_G = self.U_G[k].to(self.model.W[k].grad.device)
            U_A = self.U_A[k].to(self.model.W[k].grad.device)
            W_grad = self.model.W[k].grad.data  # already on GPU

            # diagonal scaling
            s = (U_G @ W_grad @ U_A.t()) ** 2

            # apply scaling
            delta = U_G @ W_grad @ U_A.t()
            delta = delta / (s + self.eps)

            # project back
            delta = U_G.t() @ delta @ U_A

            vec.append(delta.view(-1))
        return torch.cat(vec)

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu = False,using_native_amp=None, using_lbfgs = False):
        task_key = str(self.task_id)
        if second_order_closure is not None:
            second_order_closure()
        cur_param = parameters_to_vector(self.get_params_dict(last=False)) # pulls the model parameters into a vector
        # grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False)) # pulls the model gradients into a vector

        nat_grad_vec = self.get_natural_gradient_vector().to(self.device)
        grad_vec = nat_grad_vec
        if self.config.ogd or self.config.ogd_plus:
            # print(f"😈😈😈 Projecting the gradient vector")
            proj_grad_vec = project_vec_custom(grad_vec, proj_basis=self.ogd_basis, inner_prod=self.inner_product)
            new_grad_vec = grad_vec - proj_grad_vec
        else:
            new_grad_vec = grad_vec
        cur_param -= self.config.lr * new_grad_vec # big vector, orthogonal gradient update
        vector_to_parameters(cur_param, self.get_params_dict(last=False)) # updates the model parameters using this big vector
        optimizer.zero_grad()

    def _update_mem(self, data_train_loader, val_loader=None):
        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1

        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.config.memory_size

        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(data_train_loader.dataset[ind])

        ####################################### Grads MEM ###########################

        # (e) Get the new non-orthonormal gradients basis
        if self.config.ogd:
            print(f"😈😈😈 OGD basis retrieved")
            ogd_train_loader = torch.utils.data.DataLoader(self.task_memory[self.task_count], batch_size=1,
                                                           shuffle=False, num_workers=1)
        elif self.config.ogd_plus:
            print(f"😈😈😈 OGD_PLUS basis retrieved")
            all_task_memory = []
            for task_id, mem in self.task_memory.items():
                all_task_memory.extend(mem)
            # random.shuffle(all_task_memory)
            # ogd_memory_list = all_task_memory[:num_sample_per_task]
            ogd_memory_list = all_task_memory
            ogd_memory = Memory()
            for obs in ogd_memory_list:
                ogd_memory.append(obs)
            ogd_train_loader = torch.utils.data.DataLoader(ogd_memory, batch_size=1, shuffle=False, num_workers=1)
        
        # Non orthonormalised basis
        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader)
        print(f"new_basis_tensor shape {new_basis_tensor.shape}")

        # (f) Ortonormalise the whole memorized basis
        if self.config.is_split:
            n_params = count_parameter(self.model.linear)
        else:
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        self.ogd_basis = self.to_device(self.ogd_basis)

        if self.config.ogd :
            for t, mem in self.task_grad_memory.items():
                # Concatenate all data in each task
                task_ogd_basis_tensor = mem.get_tensor()
                task_ogd_basis_tensor = self.to_device(task_ogd_basis_tensor)
                self.ogd_basis = torch.cat([self.ogd_basis, task_ogd_basis_tensor], axis=1)
            self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)
        elif self.config.ogd_plus :
            if self.config.pca :
                for t, mem in self.task_grad_memory.items():
                    # Concatenate all data in each task
                    task_ogd_basis_tensor = mem.get_tensor()
                    task_ogd_basis_tensor = self.to_device(task_ogd_basis_tensor)

                    # task_ogd_basis_tensor.shape
                    # Out[3]: torch.Size([330762, 50])
                    start_idx = t * num_sample_per_task
                    end_idx = (t + 1) * num_sample_per_task
                    before_pca_tensor = torch.cat([task_ogd_basis_tensor, new_basis_tensor[:, start_idx:end_idx]], axis=1)
                    u, s, v = torch.svd(before_pca_tensor)

                    # u.shape
                    # Out[8]: torch.Size([330762, 150]) -> col size should be 2 * num_sample_per_task

                    after_pca_tensor = u[:, :num_sample_per_task]

                    # after_pca_tensor.shape
                    # Out[13]: torch.Size([330762, 50])

                    self.ogd_basis = torch.cat([self.ogd_basis, after_pca_tensor], axis=1)
            #   self.ogd_basis.shape should be T * num_sample_per_task

            else :
                self.ogd_basis = new_basis_tensor

        # TODO : Check if start_idx is correct :)
        start_idx = (self.task_count - 1) * num_sample_per_task
        # print(f"the start idx of orthonormalisation if {start_idx}")
        self.ogd_basis = orthonormalize_custom(self.ogd_basis, self.inner_product, normalize=True)

        # (g) Store in the new basis
        ptr = 0
        for t, mem in self.task_memory.items():
            task_mem_size = len(mem)

            idxs_list = [i + ptr for i in range(task_mem_size)]
            if self.config.gpu:
                self.ogd_basis_ids[t] = torch.LongTensor(idxs_list).cuda()
            else:
                self.ogd_basis_ids[t] = torch.LongTensor(idxs_list)

            self.task_grad_memory[t] = Memory()  # Initialize the memory slot
            for ind in range(task_mem_size):  # save it to the memory
                self.task_grad_memory[t].append(self.ogd_basis[:, ptr])
                ptr += 1
        print(f"Used memory {ptr} / {self.config.memory_size}")

        if self.config.ogd or self.config.ogd_plus :
            loader = torch.utils.data.DataLoader(self.task_memory[self.task_count],
                                                                            batch_size=self.config.batch_size,
                                                                            shuffle=True,
                                                                            num_workers=2)
            self.mem_loaders.append(loader)

    def update_ogd_basis(self, task_id, data_train_loader):
        if self.config.gpu :
            device = torch.device("cuda")
            self.model.to(device)
        print(f"\nself.model.device update_ogd_basis {next(self.model.parameters()).device}")
        if self.config.ogd or self.config.ogd_plus:
            self._update_mem(data_train_loader)
