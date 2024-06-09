import os

cudas = '4,5,7,8'
root_data_dir = '..'
dataset = 'dataset/HM'
behaviors = 'hm_50w_users.tsv'
images = 'hm_50w_items.tsv'
lmdb_data = 'hm_50w_items.lmdb'
logging_num = 4
testing_num = 1
train_emb = False
enhance = True

CV_resize = 224
CV_model_load = 'swin_base'
freeze_paras_before = 0


mode = 'train'
item_tower = 'modal'

epoch = 50
load_ckpt_name = 'None'

drop_rate_list = [0.1]
batch_size_list = [32]
embedding_dim_list = [2048]
l2_list = [(0.01, 0.01)]
lr_list = [(1e-4, 1e-4)]

for l2_flr in l2_list:
    l2_weight = l2_flr[0]
    fine_tune_l2_weight = l2_flr[1]
    for embedding_dim in embedding_dim_list:
        for batch_size in batch_size_list:
            for drop_rate in drop_rate_list:
                for lr_flr in lr_list:
                    lr = lr_flr[0]
                    fine_tune_lr = lr_flr[1]
                    label_screen = '{}_bs{}_ed{}_lr{}_dp{}_wd{}_Flr{}'.format(
                        item_tower, batch_size, embedding_dim, lr,
                        drop_rate, l2_weight, fine_tune_lr)
                    run_py = "CUDA_VISIBLE_DEVICES='{}' \
                             torchrun --nproc_per_node {} --master_port 1237\
                             run.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
                             --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                             --l2_weight {} --fine_tune_l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                             --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {} --testing".format(
                        cudas, len(cudas.split(',')),
                        root_data_dir, dataset, behaviors, images, lmdb_data,
                        mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                        l2_weight, fine_tune_l2_weight, drop_rate, batch_size, lr, embedding_dim,
                        CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr)
                    if train_emb:
                        run_py += ' --train_emb'
                    if enhance:
                        run_py += ' --enhance'
                    print(run_py)
                    os.system(run_py)
