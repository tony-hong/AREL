# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import json
import h5py
import os
import os.path
import numpy as np
import random
import logging
import misc.utils as utils

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import eval_utils

import pickle


class VISTDataset(Dataset):
    def __init__(self, opt):
        self.mode = 'train'  # by default
        self.opt = opt

        self.task = opt.task  # option: 'story_telling', 'image_captioning'
        self.data_type = {
            'whole_story': False,
            'split_story': True,
            'caption': False
        }

        print ('use_IN_feat', opt.use_IN_feat)
        print ('use_prd_feat', opt.use_prd_feat)
        print ('use_graph_emb', opt.use_graph_emb)
        
        # open the hdf5 file
        print('DataLoader loading story h5 file: ', opt.story_h5)
        self.story_h5 = h5py.File(opt.story_h5, 'r', driver='core')['story']
        print("story's max sentence length is ", self.story_h5.shape[1])

        print('DataLoader loading story h5 file: ', opt.full_story_h5)
        self.full_story_h5 = h5py.File(opt.full_story_h5, 'r', driver='core')['story']
        print("full story's max sentence length is ", self.full_story_h5.shape[1])

        print('DataLoader loading description h5 file: ', opt.desc_h5)
        self.desc_h5 = h5py.File(opt.desc_h5, 'r', driver='core')['story']
        print("caption's max sentence length is ", self.desc_h5.shape[1])

        print('DataLoader loading story_line json file: ', opt.story_line_json)
        self.story_line = json.load(open(opt.story_line_json))
        
        self.id2word = self.story_line['id2words']
        print("vocab[0] = ", self.id2word['0'])
        print("vocab[1] = ", self.id2word['1'])
        self.word2id = self.story_line['words2id']
        self.vocab_size = len(self.id2word)
        print('vocab size is ', self.vocab_size)

        self.story_ids = {'train': [], 'val': [], 'test': []}
        self.description_ids = {'train': [], 'val': [], 'test': []}
        
        story_train_set = set(self.story_line['train'].keys())
        print('Stories in training set: ', len(story_train_set))
        story_val_set = set(self.story_line['val'].keys())
        print('Stories in validation set: ', len(story_val_set))
        
        # add filtering
        if opt.story_filter_json: 
            self.story_filter = []
            if os.path.isfile(opt.story_filter_json):
                self.story_filter = json.load(open(opt.story_filter_json))
            story_filter_set = {str(idx) for idx in self.story_filter}
            print('Stories in filtered training set: ', len(story_filter_set))
            story_filtered_train_list = list(story_train_set - story_filter_set)
            print('Stories left in training set: ', len(story_filtered_train_list))
            self.story_ids['train'] = story_filtered_train_list
            story_filtered_val_list = list(story_val_set - story_filter_set)
            print('Stories left in validation set: ', len(story_filtered_val_list))
            self.story_ids['val'] = story_filtered_val_list
        else:
            self.story_ids['train'] = self.story_line['train'].keys()
            self.story_ids['val'] = self.story_line['val'].keys()
            
        self.story_ids['test'] = self.story_line['test'].keys()
        self.description_ids['train'] = self.story_line['image2caption']['train'].keys()
        self.description_ids['val'] = self.story_line['image2caption']['val'].keys()
        self.description_ids['test'] = self.story_line['image2caption']['test'].keys()

        print('There are {} training data, {} validation data, and {} test data'.format(len(self.story_ids['train']),
                                                                                        len(self.story_ids['val']),
                                                                                        len(self.story_ids['test'])))

        ref_dir = 'data/reference'
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)

        # write reference files for storytelling
        for split in ['val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                if story['album_id'] not in reference:
                    reference[story['album_id']] = [story['origin_text']]
                else:
                    reference[story['album_id']].append(story['origin_text'])
            with open(os.path.join(ref_dir, '{}_reference.json'.format(split)), 'w') as f:
                json.dump(reference, f)

        # write reference files for captioning
        for split in ['val', 'test']:
            reference = {}
            for flickr_id, story in self.story_line['image2caption_original'][split].iteritems():
                reference[flickr_id] = story
            with open(os.path.join(ref_dir, '{}_desc_reference.json'.format(split)), 'w') as f:
                json.dump(reference, f)
                
        if opt.use_prd_feat:
            # load predicate features
            train_prd_matrix_path = os.path.join(self.opt.data_dir, 'resnet_features/imgfeat2PRD_feat_2048_current_train_PRD_feat.npy')
            valid_prd_matrix_path = os.path.join(self.opt.data_dir, 'resnet_features/imgfeat2PRD_feat_2048_current_val_PRD_feat.npy')

            train_prd_dict_path = os.path.join(self.opt.data_dir, 'resnet_features/SIND_id2train_matrix_idx.pcl')
            valid_prd_dict_path = os.path.join(self.opt.data_dir, 'resnet_features/SIND_id2val_matrix_idx.pcl')

            self.prd_dict = {'train': [], 'val': [], 'test': []}
            with open(train_prd_dict_path, 'r') as f:
                self.prd_dict['train'] = pickle.load(f)
            with open(valid_prd_dict_path, 'r') as f:
                self.prd_dict['val'] = pickle.load(f)
            with open(valid_prd_dict_path, 'r') as f:
                self.prd_dict['test'] = pickle.load(f)

            self.prd_matrix = {'train': [], 'val': [], 'test': []}
            self.prd_matrix['train'] = np.load(train_prd_matrix_path)
            self.prd_matrix['val'] = np.load(valid_prd_matrix_path)
            self.prd_matrix['test'] = np.load(valid_prd_matrix_path)
        
        if self.opt.use_det:
            # load object detection features
            print('DataLoader loading story h5 file: ', opt.full_story_h5)

            objDet_train_path = os.path.join(self.opt.data_dir, 'vgg_features', 'VIST_train_obj_fmap.h5')
            objDet_valid_path = os.path.join(self.opt.data_dir, 'vgg_features', 'VIST_val_obj_fmap.h5')
            objDet_test_path = os.path.join(self.opt.data_dir, 'vgg_features', 'VIST_test_obj_fmap.h5')
            objDet_train_h5 = h5py.File(objDet_train_path, 'r')
            objDet_valid_h5 = h5py.File(objDet_valid_path, 'r')        
            objDet_test_h5 = h5py.File(objDet_test_path, 'r')

            self.objDet_dict = {'train': objDet_train_h5, 'val': objDet_valid_h5, 'test': objDet_test_h5}
            #self.objDet_dict = {'train': objDet_train_h5, 'val': objDet_valid_h5, 'test': ''}
            
        if opt.use_graph_emb:
            # load scene graph embeddings
            # change the path here
            graph_emb_list_path = os.path.join(self.opt.data_dir, 'graph_feat', self.opt.graphFeat_ver, 'graph_emb_all.npy')
            graph_emb_imgIDs_path = os.path.join(self.opt.data_dir, 'graph_feat', self.opt.graphFeat_ver, 'imgID_list_all.json')
            graphEmb_local_path = os.path.join(self.opt.data_dir, 'graph_feat', self.opt.graphFeat_ver, 'local_graphFeat_all.h5')
            
            graph_emb_list = np.load(graph_emb_list_path)
            with open(graph_emb_imgIDs_path, 'r') as f:
                graph_emb_imgIDs = json.load(f)
            graphEmb_local = h5py.File(graphEmb_local_path, 'r')

            imgID2idx = {}
            self.graphEmb_global_dict = {}
            self.graphEmb_local_dict = {}
            for idx, imgID in enumerate(graph_emb_imgIDs):
                imgID2idx[imgID] = str(idx)
                self.graphEmb_global_dict[imgID] = graph_emb_list[idx]
                self.graphEmb_local_dict[imgID] = [idx]

        # load global embeddings
        global_emb_path = os.path.join(self.opt.data_dir, 'vgg_features', 'VIST_all_avg.pkl')
        with open(global_emb_path, 'rb') as f:
            self.global_emb = pickle.load(f)
            
        
    def __getitem__(self, index):  
        use_IN_feat = self.opt.use_IN_feat
        use_prd_feat = self.opt.use_prd_feat
        use_graph_emb = self.opt.use_graph_emb

#         print ('use_IN_feat', use_IN_feat)
#         print ('use_prd_feat', use_prd_feat)
#         print ('use_graph_emb', use_graph_emb)
        
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            story = self.story_line[self.mode][story_id]

            # load feature
            feature_fc = np.zeros((story['length'], self.opt.feat_size), dtype='float32')
            feature_conv = np.zeros((story['length'], self.opt.conv_feat_size), dtype='float32')
            feature_det = np.zeros((story['length'], 64, self.opt.feat_size), dtype='float32')
            
            for i in xrange(story['length']):
                image_id = story['flickr_id'][i]
                
                '''
                # load fc feature
                ResNet_fc_path = os.path.join(self.opt.data_dir, 'resnet_features/fc', self.mode,
                                       '{}.npy'.format(image_id))
                feature_IN_ResNet_fc = np.load(ResNet_fc_path)

                double_fc = np.hstack([feature_IN_ResNet_fc, feature_IN_ResNet_fc])
                '''
                
                feature_IN_fc = self.global_emb.get(image_id, np.array([]))
                if feature_IN_fc.shape == (0,):
                    continue
                
                # change the feature here
                if use_prd_feat:
                    #print (self.mode)
                    feature_PRD_fc_idx = self.prd_dict[self.mode].get(image_id, None)
                    if feature_PRD_fc_idx:
                        feature_PRD_fc = self.prd_matrix[self.mode][feature_PRD_fc_idx]
                    elif use_IN_feat:
                        feature_PRD_fc = feature_IN_fc
                   
                if use_graph_emb:
                    feature_graphEmb_global = self.graphEmb_global_dict.get(image_id, np.array([]))
                    feature_graph_emb = feature_graphEmb_global.reshape(-1)

                    if feature_graph_emb.shape[0] > 2048:
                        feature_graph_emb = feature_graph_emb
                    elif feature_graph_emb.shape == (0,):
                        if use_IN_feat:
                            feature_graph_emb = feature_IN_fc
                        else:
                            continue
                    else:
                        SCALE_FACTOR = int(4096 / feature_graph_emb.shape[0])
                        feature_graph_emb = np.tile(feature_graph_emb, SCALE_FACTOR)
                    
                if not use_IN_feat and use_prd_feat and not use_graph_emb:
                    feature_fc[i] = np.hstack([feature_PRD_fc, feature_PRD_fc])
                elif not use_IN_feat and not use_prd_feat and use_graph_emb:
                    feature_fc[i] = feature_graph_emb
                    #feature_fc[i] = np.hstack([feature_graph_emb, feature_graph_emb])
                elif not use_IN_feat and use_prd_feat and use_graph_emb:
                    feature_fc[i] = np.hstack([feature_PRD_fc, feature_graph_emb])
                else:
                    feature_fc[i] = np.hstack([feature_IN_fc])
                
                if self.opt.use_conv:
                    conv_path = os.path.join(self.opt.data_dir, 'resnet_features/conv', self.mode,
                                             '{}.npz'.format(story['flickr_id'][i]))
                    feature_conv[i] = np.load(conv_path)['arr_0'].flatten()
                    
                if self.opt.use_det:
                    feature_VGG_det = self.objDet_dict[self.mode].get(image_id, None)
                    if not feature_VGG_det:
                        hidden_size = feature_IN_fc.shape[0]
                        feature_det[i] = np.tile(feature_IN_fc, (1, 64)).reshape(64, hidden_size)
                    elif feature_VGG_det.shape[0] != 64:
                        pad_len = 64 - feature_VGG_det.shape[0]
                        feature_det[i] = np.pad(feature_VGG_det, ((0,pad_len), (0,0)), 'constant', constant_values=(0,0))
                    else:
                        feature_det[i] = feature_VGG_det

                #print ('feature_VGG_det.shape', feature_VGG_det.shape)
                #print ('feature_det.shape', feature_det.shape)
            
            sample = {'feature_fc': feature_fc}
            
            if self.opt.use_det:
                sample['feature_det'] = feature_det

            if self.opt.use_conv:
                sample['feature_conv'] = feature_conv
            
            
            # load story
            if self.data_type['whole_story']:
                whole_story = self.full_story_h5[story['whole_text_index']]
                sample['whole_story'] = np.int64(whole_story)

            if self.data_type['split_story']:
                split_story = self.story_h5[story['text_index']]
                sample['split_story'] = np.int64(split_story)

            # load caption
            if self.data_type['caption']:
                caption = []
                for flickr_id in story['flickr_id']:
                    if flickr_id in self.story_line['image2caption'][self.mode]:
                        descriptions = self.story_line['image2caption'][self.mode][flickr_id]['caption']
                        random_idx = np.random.choice(len(descriptions), 1)[0]
                        caption.append(self.desc_h5[descriptions[random_idx]])
                    else:
                        caption.append(np.zeros((self.desc_h5.shape[1],), dtype='int64'))
                sample['caption'] = np.asarray(caption, 'int64')
            sample['index'] = np.int64(index)

            return sample

        elif self.task == "image_captioning":
            flickr_id = self.description_ids[self.mode][index]
            descriptions = self.story_line['image2caption'][self.mode][flickr_id]['caption']
            random_idx = np.random.choice(len(descriptions), 1)[0]
            description = descriptions[random_idx]

            fc_path = os.path.join(self.opt.data_dir, 'resnet_features/fc', self.mode,
                                   '{}{}.npy'.format(self.opt.prefix, flickr_id))
            conv_path = os.path.join(self.opt.data_dir, 'resnet_features/conv', self.mode, '{}.npy'.format(flickr_id))
            feature_fc = np.load(fc_path)
            feature_conv = np.load(conv_path).flatten()
            
            sample = {'feature_fc': feature_fc, 'feature_conv': feature_conv}
            target = np.int64(self.desc_h5[description])
            sample['whole_story'] = target
            sample['mask'] = np.zeros_like(target, dtype='float32')
            nonzero_num = (target != 0).sum() + 1
            sample['mask'][:nonzero_num] = 1
            sample['index'] = np.int64(index)

            return sample
        

    def __len__(self):
        if self.task == 'story_telling':
            return len(self.story_ids[self.mode])
        elif self.task == 'image_captioning':
            return len(self.description_ids[self.mode])
        else:
            raise Exception("{} task is not proper for this dataset.".format(self.task))

    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def test(self):
        self.mode = 'test'

    def set_option(self, data_type=None):
        if self.task == 'story_telling':
            if data_type is not None:
                self.data_type = data_type
        else:
            pass
            # logging.error("{} task is not proper for this dataset.".format(self.task))
            # raise Exception("{} task is not proper for this dataset.".format(self.task))

    def get_GT(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            story = self.story_line[self.mode][story_id]
            return story['origin_text']
        elif self.task == 'image_captioning':
            raise Exception("To be implemented.")
        else:
            raise Exception("{} task is not proper for this dataset.".format(self.task))

    def get_id(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            album_id = self.story_line[self.mode][story_id]['album_id']
            image_id = self.story_line[self.mode][story_id]['flickr_id']
            return album_id, image_id
        else:
            return self.description_ids[self.mode][index]

    def get_all_id(self, index):
        story_id = self.story_ids[self.mode][index]
        album_id = self.story_line[self.mode][story_id]['album_id']
        image_id = self.story_line[self.mode][story_id]['flickr_id']
        return story_id, album_id, image_id

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.id2word

    def get_word2id(self):
        return self.word2id

    def get_whole_story_length(self):
        return self.full_story_h5.shape[1]

    def get_story_length(self):
        return self.story_h5.shape[1]

    def get_caption_length(self):
        return self.desc_h5.shape[1]


if __name__ == "__main__":
    import sys
    import os
    import opts
    import time

    start = time.time()

    opt = opts.parse_opt()
    dataset = VISTDataset(opt)

    print("dataset finished: ", time.time() - start)
    start = time.time()

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=opt.shuffle, num_workers=8)

    print("dataloader finished: ", time.time() - start)

    dataset.train()  # train() mode has to be called before using train_loader
    for iter, batch in enumerate(train_loader):
        print("enumerate: ", time.time() - start)
        print(iter)
        print(batch['feature_fc'].shape)
        #print(batch[0].size())
