from ogd.tools import *
import pytorch_lightning as pl


class AgentModel(pl.LightningModule):
    def __init__(self, config, wandb_run, val_loaders):
        super().__init__()
        self.criterion_fn = nn.CrossEntropyLoss()
        self.task_id = 0 ## YY: initialized task_id

        self.config = config
        self.wandb_run = wandb_run

        self.model = self.create_model()
        print(f"### The model is {self.model} ###")
        print(f"### The model has {count_parameter(self.model)} parameters ###")

        # # TODO : remove from init : added only for the NTK gen part ?
        # self.optimizer = self.optimizer = torch.optim.SGD(params=self.model.parameters(),
        #                                                   lr=self.config.lr,
        #                                                   momentum=0,
        #                                                   weight_decay=0)

        if self.config.is_split_cub :
            n_params = get_n_trainable(self.model)
        elif self.config.is_split :
            n_params = count_parameter(self.model.linear)
        else :
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

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        # in_channel = 1 i f self.config.dataset == "MNIST" else 3

        # if cfg.model_type not in ["alexnet"] :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # if cfg.model_type not in ["mlp", "lenet"] :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # else :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim,
        #                                                                            dropout=self.config.dropout)
                                                                               # in_channel=in_channel)
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim)

        n_feat = model.last.in_features
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)

        if self.config.gpu :
            device = torch.device("cuda")
            model.to(device)
        return model

    def _get_new_ogd_basis(self, train_loader, last=False):
        return self._get_neural_tangents(train_loader,
                                         gpu=self.config.gpu,
                                         optimizer=self.optimizer,
                                         model=self.model, last=last)

    def _get_neural_tangents(self, train_loader, gpu, optimizer, model, last):
        new_basis = []

        for i, (inputs, targets, tasks) in tqdm(enumerate(train_loader),
                                                desc="get neural tangents",
                                                total=len(train_loader.dataset)):
            # if gpu:
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)

            out = self.forward(x=inputs, task=(tasks))
            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=last))
            new_basis.append(grad_vec)
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    def forward(self, x, task):
        task_key = task[0]
        out = self.model.forward(x)
        if self.config.is_split :
            return out[task_key]
        else :
            return out["All"]

    def predict(self, x, task):
        x = torch.tensor(x)
        x = self.to_device(x)
        out = self.forward(x, task)
        _, pred = out.topk(1, 1, True, True)
        return pred

    def training_step(self, batch, batch_nb):
        self.model.train()
        assert self.model.training

        data, target, task = batch
        output = self.forward(data, task)
        loss = self.criterion_fn(output, target)

        self.task_id = int(task[0])
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
            # wandb.log(log_dict,
            #           commit=False)
            wandb.log(log_dict)

        # wandb.log({f"train_loss_{self.task_id}": loss,
        #            "task_id": self.task_id})

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

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu = False,using_native_amp=None, using_lbfgs = False):
        task_key = str(self.task_id)
        if second_order_closure is not None:
            second_order_closure()

        cur_param = parameters_to_vector(self.get_params_dict(last=False))
        grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
        if self.config.ogd or self.config.ogd_plus:
            # print(f"😈😈😈 Projecting the gradient vector")
            proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis, gpu=self.config.gpu)
            new_grad_vec = grad_vec - proj_grad_vec
        else:
            new_grad_vec = grad_vec
        cur_param -= self.config.lr * new_grad_vec
        vector_to_parameters(cur_param, self.get_params_dict(last=False))

        if self.config.is_split :
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True, task_key=task_key))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, task_key=task_key))
            cur_param -= self.config.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True, task_key=task_key))
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
        self.ogd_basis = orthonormalize(self.ogd_basis, gpu=self.config.gpu, normalize=True)

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


class AgentModel_ONG(pl.LightningModule):
    def __init__(self, config, wandb_run, val_loaders):
        print("✅✅✅✅ INITIALZING ONG AGENTMODEL")
        super().__init__()
        self.criterion_fn = nn.CrossEntropyLoss()

        self.config = config
        self.wandb_run = wandb_run

        self.task_id = 0  ## TODO(YY): Initialize task_id
        self.eps = 1e-3
        self.eigbasis_comp_freq = 20
        self.ekfac_step = 0
        self.ekfac_A = []
        self.ekfac_G = []
        self.ekfac_U_A = []
        self.ekfac_U_G = []

        self.model = self.create_model()
        self._register_ekfac_hooks()
        print(f"### The model has {count_parameter(self.model)} parameters ###")
        self.ekfac_params = []
        for p in self.get_params_dict(last=False):
            if p.dim() == 2:
                self.ekfac_params.append(p)
        # now build the A, G, U_A, U_G buffers in the same order
        for W in self.ekfac_params:
            self.ekfac_A.append(torch.eye(W.size(1), device=W.device))
            self.ekfac_G.append(torch.eye(W.size(0), device=W.device))
            self.ekfac_U_A.append(torch.eye(W.size(1), device=W.device))
            self.ekfac_U_G.append(torch.eye(W.size(0), device=W.device))

        # # TODO : remove from init : added only for the NTK gen part ?
        # self.optimizer = self.optimizer = torch.optim.SGD(params=self.model.parameters(),
        #                                                   lr=self.config.lr,
        #                                                   momentum=0,
        #                                                   weight_decay=0)

        if self.config.is_split_cub :
            n_params = get_n_trainable(self.model)
        elif self.config.is_split :
            n_params = count_parameter(self.model.linear)
        else :
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
    
    def get_nat_grad_vec(self, all_params):
        self.ekfac_step += 1
        if (self.ekfac_step - 1) % self.eigbasis_comp_freq == 0:
            self._update_ekfac_stats()

        # 1) get the full gradient vector
        full_grad_vec = parameters_to_grad_vector(all_params)

        # 2) unpack it and build nat‑grad per‑param
        offset = 0
        nat_parts = []
        # we'll also need a pointer into the EKFAC lists
        ek = 0

        for p in all_params:
            numel = p.numel()
            grad_chunk = full_grad_vec[offset:offset+numel].view_as(p)
            offset += numel

            if p.dim() != 2:
                # biases, etc.: just pass gradient unchanged
                nat_parts.append(grad_chunk.view(-1))
                continue

            # now we know p==self.ekfac_params[ek]
            assert p is self.ekfac_params[ek], \
                f"Param order mismatch at ek={ek}"
            U_G = self.ekfac_U_G[ek]
            U_A = self.ekfac_U_A[ek]
            # sanity check
            if grad_chunk.shape != (U_G.shape[0], U_A.shape[0]):
                raise RuntimeError(
                    f"shape mismatch for layer {ek}: grad {tuple(grad_chunk.shape)}, "
                    f"U_G {tuple(U_G.shape)}, U_A {tuple(U_A.shape)}"
                )

            # natural gradient in eigenbasis
            delta = U_G @ grad_chunk @ U_A.T
            delta = delta / (delta**2 + self.eps)
            nat = U_G.T @ delta @ U_A

            nat_parts.append(nat.view(-1))
            ek += 1
        return torch.cat(nat_parts)
    
    def _update_ekfac_stats(self):
        # Use one memory batch to estimate covariances
        loader = self.mem_loaders[-1] if self.mem_loaders else None
        if loader is None:
            return

        data_iter = iter(loader)
        inputs, targets, tasks = next(data_iter)
        inputs = self.to_device(inputs)
        targets = self.to_device(targets)

        # Zero out existing gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Forward pass
        out = self.forward(inputs, task=tasks)  # shape [B, C]
        B = out.shape[0]
        # Gather the logits corresponding to each true label
        idx = torch.arange(B, device=out.device)
        logits = out[idx, targets]           # shape [B]
        loss = logits.sum()                  # scalar sum over batch
        loss.backward()

        # Build new covariance estimates
        modules = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        A_, G_ = [], []
        for m in modules:
            A_in = m.input               # [B, in_features]
            G_out = m.output.grad        # [B, out_features]
            A_.append((A_in.T @ A_in) / A_in.shape[0])
            G_.append((G_out.T @ G_out) / G_out.shape[0])

        # Update running averages and recompute eigenbases
        rho = min(1 - 1 / self.ekfac_step, 0.95)
        for k in range(len(self.ekfac_A)):
            self.ekfac_A[k] = rho * self.ekfac_A[k] + (1 - rho) * A_[k].detach()
            self.ekfac_G[k] = rho * self.ekfac_G[k] + (1 - rho) * G_[k].detach()
            _, self.ekfac_U_A[k] = torch.linalg.eigh(self.ekfac_A[k])
            _, self.ekfac_U_G[k] = torch.linalg.eigh(self.ekfac_G[k])

    def _register_ekfac_hooks(self):
        def forward_hook(module, input, output):
            module.input = input[0].detach()
            if output.requires_grad:
                output.retain_grad()  # To get gradient in backward
            module.output = output  # We access output.grad during EKFAC update

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(forward_hook)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        # in_channel = 1 i f self.config.dataset == "MNIST" else 3

        # if cfg.model_type not in ["alexnet"] :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # if cfg.model_type not in ["mlp", "lenet"] :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # else :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim,
        #                                                                            dropout=self.config.dropout)
                                                                               # in_channel=in_channel)
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim)

        n_feat = model.last.in_features
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)

        if self.config.gpu :
            device = torch.device("cuda")
            model.to(device)
        return model

    def _get_new_ogd_basis(self, train_loader, last=False):
        return self._get_neural_tangents(train_loader,
                                         gpu=self.config.gpu,
                                         optimizer=self.optimizer,
                                         model=self.model, last=last)

    def _get_neural_tangents(self, train_loader, gpu, optimizer, model, last):
        new_basis = []

        for i, (inputs, targets, tasks) in tqdm(enumerate(train_loader),
                                                desc="get neural tangents",
                                                total=len(train_loader.dataset)):
            # if gpu:
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)

            out = self.forward(x=inputs, task=(tasks))
            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            # grad_vec = parameters_to_grad_vector(self.get_params_dict(last=last))
            nat_grad_vec = self.get_nat_grad_vec(list(self.get_params_dict(last=False)))
            grad_vec = nat_grad_vec
            new_basis.append(grad_vec)
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    def forward(self, x, task):
        task_key = task[0]
        out = self.model.forward(x)
        if self.config.is_split :
            return out[task_key]
        else :
            return out["All"]

    def predict(self, x, task):
        x = torch.tensor(x)
        x = self.to_device(x)
        out = self.forward(x, task)
        _, pred = out.topk(1, 1, True, True)
        return pred

    def training_step(self, batch, batch_nb):
        self.model.train()
        assert self.model.training

        data, target, task = batch
        output = self.forward(data, task)
        loss = self.criterion_fn(output, target)

        self.task_id = int(task[0])
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
            # wandb.log(log_dict,
            #           commit=False)
            wandb.log(log_dict)

        # wandb.log({f"train_loss_{self.task_id}": loss,
        #            "task_id": self.task_id})

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

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure=None, using_native_amp=None, **kwargs):
        task_key = str(self.task_id)
        if optimizer_closure is not None:
            optimizer_closure()
        else:
            return

        cur_param = parameters_to_vector(self.get_params_dict(last=False))
        ## TODO(YY): us projection, nat_grad, reproject here to compute the natural_grad_vec
        nat_grad_vec = self.get_nat_grad_vec(list(self.get_params_dict(last=False)))
        grad_vec = nat_grad_vec
        if grad_vec is None:
            raise ValueError("grad_vec is None")
        
        ## UNCOMMENT THIS LINE TO GO BACK TO NORMAL GRADIENT CALCULATION
        # grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))

        if self.config.ogd or self.config.ogd_plus:
            proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis, gpu=self.config.gpu)
            new_grad_vec = grad_vec - proj_grad_vec
        else:
            new_grad_vec = grad_vec
        cur_param -= self.config.lr * new_grad_vec
        vector_to_parameters(cur_param, self.get_params_dict(last=False))

        if self.config.is_split :
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True, task_key=task_key))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, task_key=task_key))
            cur_param -= self.config.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True, task_key=task_key))
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
            print(f"😈😈😈 OGD basis retrieving")
            ogd_train_loader = torch.utils.data.DataLoader(self.task_memory[self.task_count], batch_size=1,
                                                           shuffle=False, num_workers=1)
        elif self.config.ogd_plus:
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
        self.ogd_basis = orthonormalize(self.ogd_basis, gpu=self.config.gpu, normalize=True)

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


class AgentModel_ONG_2(pl.LightningModule):
    def __init__(self, config, wandb_run, val_loaders):
        print("🔥🔥🔥 Initializing ONG 2")
        super().__init__()
        self.config = config
        self.wandb_run = wandb_run
        self.criterion_fn = nn.CrossEntropyLoss()
        self.task_id = 0

        # Build model
        self.model = self.create_model()
        n_params = (
            get_n_trainable(self.model) if config.is_split_cub else
            count_parameter(self.model.linear) if config.is_split else
            count_parameter(self.model)
        )

        # OGD basis storage
        self.ogd_basis = torch.empty(n_params, 0)
        if config.gpu:
            self.ogd_basis = self.ogd_basis.cuda()
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]))
        if config.gpu:
            self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())

        # EKFAC / Natural gradient state
        self.eps = 1e-3
        # self.eigbasis_comp_freq = getattr(config, 'ekfac_freq', 20)
        self.eigbasis_comp_freq = 20
        self.ekfac_step = 0
        # Identify 2D weights to approximate
        self.ekfac_params = []
        for p in self.get_params_dict(last=False):
            if p.dim() == 2:
                self.ekfac_params.append(p)
        # Initialize covariance and eigen-basis
        self.ekfac_A = [torch.eye(p.size(1), device=p.device) for p in self.ekfac_params]
        self.ekfac_G = [torch.eye(p.size(0), device=p.device) for p in self.ekfac_params]
        self.ekfac_U_A = [A.clone() for A in self.ekfac_A]
        self.ekfac_U_G = [G.clone() for G in self.ekfac_G]
        # Hook for EKFAC
        self.mem_loaders = []
        self._register_ekfac_hooks()

        # Memory loaders & storage
        self.val_loaders = val_loaders
        self.task_memory = {}
        self.task_grad_memory = {}
        self.task_count = 0

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim)

        n_feat = model.last.in_features
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)

        if self.config.gpu:
            device = torch.device("cuda")
            model.to(device)
        return model

    def forward(self, x, task):
        task_key = task[0]
        out = self.model.forward(x)
        if self.config.is_split :
            return out[task_key]
        else :
            return out["All"]

    def get_params_dict(self, last, task_key=None):
        if self.config.is_split_cub:
            return (self.model.fc[task_key].parameters() if last else self.model.features.parameters())
        if self.config.is_split:
            return (self.model.fc[task_key].parameters() if last else self.model.linear.parameters())
        return self.model.parameters()

    def to_device(self, x):
        return x.cuda() if self.config.gpu else x

    # --- EKFAC Helpers ----------------------------------------------------------------
    def _register_ekfac_hooks(self):
        def forward_hook(module, inp, outp):
            module._inp = inp[0].detach().cuda() if self.config.gpu else inp[0].detach()
            if isinstance(outp, torch.Tensor) and outp.requires_grad:
                outp.retain_grad()
            module._out = outp
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_hook(forward_hook)

    def _update_ekfac_stats(self):
        # Estimate covariances on the most recent memory loader
        if not self.mem_loaders:
            return
        loader = self.mem_loaders[-1]
        data_iter = iter(loader)
        inputs, targets, tasks = next(data_iter)
        inputs, targets = self.to_device(inputs), self.to_device(targets)

        # zero grads
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # forward/backward on memory batch
        out = self.forward(inputs, tasks)
        B = out.shape[0]
        idx = torch.arange(B, device=out.device)
        logits = out[idx, targets]
        loss = logits.sum()
        loss.backward()

        # collect new covariances
        modules = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        A_list, G_list = [], []
        for m in modules:
            A_in = m._inp       # shape [B, in_dim]
            G_out = m._out.grad # shape [B, out_dim]
            A_list.append((A_in.T @ A_in) / A_in.shape[0])
            G_list.append((G_out.T @ G_out) / G_out.shape[0])

        # moving average and eigen-decomp
        rho = min(1 - 1 / self.ekfac_step, 0.95)
        eps = 1e-6
        for k in range(len(self.ekfac_A)):
            self.ekfac_A[k] = rho * self.ekfac_A[k] + (1 - rho) * A_list[k].detach()
            self.ekfac_G[k] = rho * self.ekfac_G[k] + (1 - rho) * G_list[k].detach()
            A = self.ekfac_A[k]
            G = self.ekfac_G[k]
            A += eps * torch.eye(A.shape[0], device=A.device)
            G += eps * torch.eye(G.shape[0], device=G.device)
            _, self.ekfac_U_A[k] = torch.linalg.eigh(A)
            _, self.ekfac_U_G[k] = torch.linalg.eigh(G)

    def get_nat_grad_vec(self, raw_grad_vec):
        # update basis periodically
        self.ekfac_step += 1
        if (self.ekfac_step - 1) % self.eigbasis_comp_freq == 0:
            self._update_ekfac_stats()

        parts, offset, ek = [], 0, 0
        for p in self.get_params_dict(last=False):
            n = p.numel()
            chunk = raw_grad_vec[offset:offset+n].view_as(p)
            offset += n
            if p.dim() != 2:
                parts.append(chunk.view(-1))
            else:
                U_G, U_A = self.ekfac_U_G[ek], self.ekfac_U_A[ek]
                delta = U_G @ chunk @ U_A.T
                delta = delta / (delta**2 + self.eps)
                nat = U_G.T @ delta @ U_A
                parts.append(nat.view(-1))
                ek += 1
        return torch.cat(parts)

    # --- Training / Optimization ------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y, task = batch
        x, y = self.to_device(x), self.to_device(y)
        out = self.forward(x, task)
        return self.criterion_fn(out, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.config.lr)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure=None, **kwargs):
        if optimizer_closure is not None:
            optimizer_closure()
        # raw & natural grad
        raw = parameters_to_grad_vector(self.get_params_dict(last=False))
        nat = self.get_nat_grad_vec(raw)
        # OGD projection
        if self.config.ogd or self.config.ogd_plus:
            proj = project_vec(nat, proj_basis=self.ogd_basis, gpu=self.config.gpu)
            final = nat - proj
        else:
            final = nat
        # update params
        vec = parameters_to_vector(self.get_params_dict(last=False))
        vec -= self.config.lr * final
        vector_to_parameters(vec, self.get_params_dict(last=False))
        optimizer.zero_grad()

    # --- OGD basis construction -----------------------------------------------------
    def _get_neural_tangents(self, train_loader, gpu, optimizer, model, last):
        new_basis = []
        for x, y, task in train_loader:
            x, y = self.to_device(x), self.to_device(y)
            out = self.forward(x, task)
            pred = out[0, y.item()]
            optimizer.zero_grad()
            pred.backward()
            raw = parameters_to_grad_vector(self.get_params_dict(last=last))
            nat = self.get_nat_grad_vec(raw)
            new_basis.append(nat)
        return torch.stack(new_basis).T

    def _get_new_ogd_basis(self, train_loader, last=False):
        opt = self.trainer.optimizers[0]
        return self._get_neural_tangents(train_loader, self.config.gpu, opt, self.model, last)

    def _update_mem(self, data_train_loader, _=None):
        self.task_count += 1
        # build memory loader (omitted)
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.config.memory_size

        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(data_train_loader.dataset[ind])

        mem_loader = torch.utils.data.DataLoader(self.task_memory[self.task_count], batch_size=1)
        self.mem_loaders.append(mem_loader)
        new_basis = self._get_new_ogd_basis(mem_loader)
        self.ogd_basis = orthonormalize(
            torch.cat([self.ogd_basis, new_basis], dim=1),
            gpu=self.config.gpu, normalize=True
        )

    def update_ogd_basis(self, task_id, data_train_loader):
        if self.config.gpu :
            device = torch.device("cuda")
            self.model.to(device)
        if self.config.ogd or self.config.ogd_plus:
            self._update_mem(data_train_loader)