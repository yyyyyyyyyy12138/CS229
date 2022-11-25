import torch
import wandb


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.num_epochs = args.epochs
        self.log_freq = args.log_freq
        self.log_cnt = args.log_cnt
        self.ckpt_freq = args.ckpt_freq
        self.ckpt_path = args.ckpt_path
        self.ckpt_load = args.ckpt_load
        wandb.init(config=args)

    def fit(self):
        start_epoch = 0

        # if load model and optimizer from prev checkpoint
        if self.ckpt_load:
            ckpt = torch.load(self.ckpt_path)
            start_epoch = ckpt['epoch']  # start from prev epoch
            self.model.net.load_state_dict(ckpt['model_state_dict'])
            self.model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # train and test model
        for i in range(start_epoch, self.num_epochs):
            self.train_epoch(i)
            self.test_epoch(i)

            # save checkpoints per ckpt frequency
            if i % self.ckpt_freq == 0:
                # TODO: save lr scheduler
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.model.net.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                }, self.ckpt_path)

    def train_epoch(self, epoch):
        accs = []
        batch_sizes = []
        # training
        self.model.net.train()  # switch to train mode
        for i, data in enumerate(self.data.training_loader):
            # pass data to device
            inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)
            # Zero your gradients for every batch!
            self.model.optimizer.zero_grad()

            # performs an inference, get predictions from the model of an input batch
            logits = self.model.net(inputs)
            # get predicted class
            preds = torch.argmax(logits, dim=1)

            # calculate accuracy
            acc = torch.sum(preds == labels) / preds.shape[0]
            accs.append(acc)
            batch_sizes.append(labels.shape[0])
            # Compute the loss and its gradients
            loss = self.model.criterion(logits, labels)
            loss.backward()
            if i % self.log_freq == 0:
                # print and log loss and accuracy per log_freq iterations
                print(f"[Train] Epoch {epoch} iteration{i}: loss={loss}, accuracy={acc}")
                wandb.log({"Train Loss": loss, "Train Accuracy": acc})
            # Adjust learning weights
            self.model.optimizer.step()

        # learning rate decay scheduler
        self.model.scheduler.step()
        accs = torch.Tensor(accs)
        # calculate epoch accuracy (weighted average)
        batch_sizes = torch.Tensor(batch_sizes)
        epoch_acc = (torch.dot(accs, batch_sizes)).item() / torch.sum(batch_sizes)
        epoch_acc = epoch_acc.item()
        print(f"[Train] Epoch {epoch} accuracy={epoch_acc}")
        # log learning rate per epoch
        param_groups = self.model.optimizer.param_groups
        for param_group in param_groups:
            wandb.log({"Learning Rate": param_group["lr"]})

    def test_epoch(self, epoch):
        # testing current epoch
        accs = []
        batch_sizes = []
        self.model.net.eval()  # switch to test mode
        with torch.no_grad():
            # log_img_freq = len(self.data.test_loader) // self.log_cnt
            for i, data in enumerate(self.data.test_loader):
                inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)

                # performs an inference, get predictions from the model of an input batch
                logits = self.model.net(inputs)
                preds = torch.argmax(logits, dim=1)  # get predicted class

                # log image in W&B (10 images per epoch)
                # if i % log_img_freq == 0:
                #     my_table = wandb.Table()
                #     import numpy as np
                #     image = np.moveaxis(inputs[0].cpu().numpy(), 0, -1)
                #     my_table.add_column("image", image)
                #     my_table.add_column("label", labels[0])
                #     my_table.add_column("class_prediction", preds[0])
                #     wandb.log({"Baseline_predictions": my_table})

                # calculate accuracy
                acc = torch.sum(preds == labels) / preds.shape[0]
                accs.append(acc)
                batch_sizes.append(labels.shape[0])

            accs = torch.Tensor(accs)
            batch_sizes = torch.Tensor(batch_sizes)
            epoch_acc = (torch.dot(accs, batch_sizes)).item() / torch.sum(batch_sizes)
            epoch_acc = epoch_acc.item()
            print(f"[Test] Epoch {epoch} accuracy={epoch_acc}")
            # log test accuracy per epoch
            wandb.log({"Test Accuracy": epoch_acc})
