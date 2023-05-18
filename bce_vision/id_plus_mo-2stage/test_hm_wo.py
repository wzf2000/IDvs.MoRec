import os

root_data_dir = '../../'
dataset = 'Dataset/Hm-large'
behaviors = 'hm_50w_users.tsv'
images = 'hm_50w_items.tsv'
lmdb_data = 'hm_50w_items.lmdb'
logging_num = 4
testing_num = 1

CV_resize = 224
CV_model_load = 'swin_tiny'
freeze_paras_before = 0


mode = 'train'
item_tower_list = ['modal_add', 'modal_cat']

epoch = 150
load_ckpt_name_list = ['epoch-48.pt', 'epoch-8.pt']


l2_weight_list = [0.01]
drop_rate_list = [0.1]
batch_size_list = [64]
lr_list_ct = [(1e-4, 0)]

embedding_dim_list = [512]

mo_dnn_layers_list = [8, 8]
dnn_layers_list = [4, 0]


for l2_weight in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for embedding_dim in embedding_dim_list:
                for lr_ct in lr_list_ct:
                    lr = lr_ct[0]
                    fine_tune_lr = lr_ct[1]
                    for i in range(len(load_ckpt_name_list)):
                        load_ckpt_name = load_ckpt_name_list[i]
                        dnn_layers = dnn_layers_list[i]
                        mo_dnn_layers = mo_dnn_layers_list[i]
                        item_tower = item_tower_list[i]
                        label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}'.format(
                            item_tower, batch_size, embedding_dim, lr,
                            drop_rate, l2_weight, fine_tune_lr)
                        run_py = "CUDA_VISIBLE_DEVICES='0' \
                                 /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
                                 run_test.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
                                 --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                 --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                 --CV_resize {} --CV_model_load {}  --epoch {}\
                                 --freeze_paras_before {}  --fine_tune_lr {}\
                                 --mo_dnn_layers {} --dnn_layers {}".format(
                            root_data_dir, dataset, behaviors, images, lmdb_data,
                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                            l2_weight, drop_rate, batch_size, lr, embedding_dim,
                            CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr,
                            mo_dnn_layers, dnn_layers)
                        os.system(run_py)
