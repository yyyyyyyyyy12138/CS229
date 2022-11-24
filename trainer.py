import torch


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.num_epochs = args.epochs

    def fit(self):
        for i in range(self.num_epochs):
            self.train_epoch(i)
            self.test_epoch(i)

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
            if i % 10 == 0:
                print(f"[Train] Epoch {epoch} iteration{i}: loss={loss}, accuracy={acc}")
            # Adjust learning weights
            self.model.optimizer.step()
        accs = torch.Tensor(accs)
        # calculate epoch accuracy (weighted average)
        batch_sizes = torch.Tensor(batch_sizes)
        epoch_acc = (torch.dot(accs, batch_sizes)).item() / torch.sum(batch_sizes)
        epoch_acc = epoch_acc.item()
        print(f"[Train] Epoch {epoch} accuracy={epoch_acc}")

    def test_epoch(self, epoch):
        # testing current epoch
        accs = []
        batch_sizes = []
        self.model.net.eval()  # switch to test mode
        with torch.no_grad():
            for k, data in enumerate(self.data.test_loader):
                inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)

                # performs an inference, get predictions from the model of an input batch
                logits = self.model.net(inputs)
                preds = torch.argmax(logits, dim=1)  # get predicted class

                # calculate accuracy
                acc = torch.sum(preds == labels) / preds.shape[0]
                accs.append(acc)
                batch_sizes.append(labels.shape[0])
                # Adjust learning weights
            accs = torch.Tensor(accs)
            batch_sizes = torch.Tensor(batch_sizes)
            epoch_acc = (torch.dot(accs, batch_sizes)).item() / torch.sum(batch_sizes)
            epoch_acc = epoch_acc.item()
            print(f"[Test] Epoch {epoch} accuracy={epoch_acc}")
