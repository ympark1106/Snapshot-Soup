from data import cifar10, cifar100, ham10000, eyepacs


# Data loader setup
def setup_data_loaders(args, data_path, batch_size):
    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
        valid_loader = None
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_loader, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=batch_size, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    # elif args.data == 'bloodmnist':
    #     _, valid_loader, test_loader = bloodmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    # elif args.data == 'pathmnist':
    #     _, valid_loader, test_loader = pathmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    # elif args.data == 'retinamnist':
    #     _, valid_loader, test_loader = retinamnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    elif args.data == 'eyepacs':
        train_loader, valid_loader, test_loader = eyepacs.get_dataloaders(data_path, batch_size=batch_size, pin_memory=True, num_workers=16)
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    return train_loader, valid_loader, test_loader