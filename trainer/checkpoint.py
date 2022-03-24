import torch

def save(name, epoch, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, name) 
    
def load(args, model, optimizer):
    if args.load_model:
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = None
    
    return epoch

