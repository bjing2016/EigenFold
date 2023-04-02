from utils.parsing import parse_train_args
args = parse_train_args()
import os, yaml, torch, wandb
import numpy as np
from model import get_model
from utils.dataset import get_loader
from utils.training import get_optimizer, get_scheduler, epoch, save_yaml_file, save_loss_plot
import pandas as pd

from utils.logging import get_logger
logger = get_logger(__name__)

def load_args(args):
    epochs, resume, commit, wandb = args.epochs, args.resume, args.commit, args.wandb
    f = open(f'{args.resume}/args.yaml')
    args.__dict__.update(yaml.full_load(f))
    args.epochs, args.resume, args.commit, args.wandb = epochs, resume, commit, wandb
    return args

if args.resume: args = load_args(args)

def wandb_init():
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(
        entity=os.environ['WANDB_ENTITY'],
        settings=wandb.Settings(start_method="fork"),
        project=args.wandb,
        name=str(args.time),
        config=args
    )
    
def main():
    if args.resume:
        logger.info(f'Resuming run with ID {args.time} and commit {args.commit}')
    else:
        logger.info(f'Initializing run with ID {args.time} and commit {args.commit}')
    
    if args.wandb: wandb_init()

    logger.info(f'Loading splits {args.splits}')
    splits = pd.read_csv(args.splits)
    try: splits = splits.set_index('path')   
    except: splits = splits.set_index('name')   
    pyg_data = None
    
    logger.info("Constructing model")
    model = get_model(args)
    
    enn_numel = sum([p.numel() for p in model.enn.parameters()])
    logger.info(f"ENN has {enn_numel} params")
    if args.wandb:
        wandb.log({'enn_numel': enn_numel})

    if not args.dry_run:
        model_dir = os.path.join(args.workdir, str(args.time))
    else:
        model_dir = os.path.join(args.workdir, 'dry_run')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    
    if not args.resume: 
        yaml_file_name = os.path.join(model_dir, 'args.yaml')
        save_yaml_file(yaml_file_name, args.__dict__)
        logger.info(f"Saving training args to {yaml_file_name}")
        
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    train_loader = get_loader(
        args, pyg_data, splits, mode='train', shuffle=True
    )
    val_loader = get_loader(
        args, pyg_data, splits, mode='val', shuffle=False
    )
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    ### DOESN'T WORK WITH DISTRIBUTED YET ###
    if args.resume:
        logger.info(f"Loading state dict {args.resume}/last_model.pt")
        state_dict = torch.load(f'{args.resume}/last_model.pt', 
                        map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        run_training(args, model, optimizer, scheduler,
            train_loader, val_loader, device, model_dir=model_dir,
            ep = state_dict['epoch']+1,
            best_val_loss=state_dict['best_val_loss'],
            best_epoch=state_dict['best_epoch'])
    
 
    run_training(args, model, optimizer, scheduler,
              train_loader, val_loader, device, model_dir=model_dir)

def gather_log(log):
    log_list = hvd.allgather_object(log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log
        
def run_training(args, model, optimizer, scheduler,
                train_loader, val_loader, device, model_dir=None, 
                ep=1, best_val_loss = np.inf, best_epoch = 1):
    
    while ep <= args.epochs:
        
        logger.info(f"Starting training epoch {ep}")
        log = epoch(args, model, train_loader, optimizer=optimizer, scheduler=scheduler,
                    device=device, print_freq=args.print_freq)

        train_loss, train_base_loss = np.nanmean(log['loss']), np.nanmean(log['base_loss'])
        logger.info(
            f"Train epoch {ep}: len {len(log['loss'])} loss {train_loss}  base loss {train_base_loss}"
        )
        
        logger.info(f"Starting validation epoch {ep}")
        log = epoch(args, model, val_loader, device=device, print_freq=args.print_freq)
        
        val_loss, val_base_loss = np.nanmean(log['loss']), np.nanmean(log['base_loss'])
        logger.info(f"Val epoch {ep}: len {len(log['loss'])} loss {val_loss}  base loss {val_base_loss}")

        ### Save val loss plot
        png_path = os.path.join(model_dir, str(ep) + '.png')
        save_loss_plot(log, png_path)
        csv_path = os.path.join(model_dir, str(ep) + '.val.csv')
        pd.DataFrame(log).to_csv(csv_path)
        logger.info(f"Saved loss plot {png_path} and csv {csv_path}")

        ### Check if best epoch
        new_best = False
        if val_loss <= best_val_loss:
            best_val_loss = val_loss; best_epoch = ep
            logger.info(f"New best val epoch")
            new_best = True

        ### Save checkpoints
        state = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epoch': ep,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }
        if new_best:
            path = os.path.join(model_dir, 'best_model.pt')
            logger.info(f"Saving best checkpoint {path}")
            torch.save(state, path)

        if ep % args.save_freq == 0:
            path = os.path.join(model_dir, f'epoch_{ep}.pt')
            logger.info(f"Saving epoch checkpoint {path}")
            torch.save(state, path)

        path = os.path.join(model_dir, 'last_model.pt')
        logger.info(f"Saving last checkpoint {path}")
        torch.save(state, path)
        
        ### Update WANDB
        update = {
            'train_loss': train_loss,
            'train_base_loss': train_base_loss,
            'val_loss': val_loss,
            'val_base_loss': val_base_loss,
            'current_lr': scheduler.get_last_lr()[0],
            'epoch': ep
        }
        logger.info(str(update))
        if args.wandb: wandb.log(update)
            
        ep += 1
        
    logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    
    
if __name__ == '__main__':
    main()