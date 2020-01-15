
# coding: utf-8

import numpy as np
import os
import pickle
import networkx as nx
from matplotlib import pyplot as plt
import argparse

def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Parser for location phrase generation')
    parser.add_argument('--input_pred_dir', type = str, 
                        default = '/data/zekunl/text_linking/pred_scripts/predictions_v1_20_od/')
    parser.add_argument('--vision_api_result_dir', type = str, 
                        default = '/data/zekunl/Geolocalizer/google_grouping/')
    parser.add_argument('--IF_WCC', action = 'store_true')
    parser.add_argument('--output_phrases_dir', type = str, default = 'phrases_v1_20_od/')
    return (parser.parse_args())


# remove empty prediction
def remove_empty_neighbors(predictions):
    preds = {}
    for anchor_idx, pred in predictions.items():
        pred = list(pred)
        if anchor_idx in pred:
            pred.remove(anchor_idx)
        if len(pred) != 0:
            preds[anchor_idx] = pred
    return preds

# remove itself from prediction 
def remove_self_neighbors(predictions):
    preds = {}
    for anchor_idx, pred in predictions.items():
        pred = list(pred)
        if anchor_idx in pred:
            pred.remove(anchor_idx)
        
        preds[anchor_idx] = pred
    return preds


def construct_graph_cc_from_neighbor_dict(neighbor_dict, IF_WCC):
    #neighbors = remove_empty_neighbors(neighbor_dict) # neighbor is still a dictionary with empty neighbors removed
    neighbors = remove_self_neighbors(neighbor_dict) # dictionary with itself removed from neighbor list
    #neighbors = neighbor_dict
    
    
    G = nx.DiGraph()
    G.add_nodes_from(neighbors.keys())
    
    #print neighbors.keys()
    for anchor_idx, nei in neighbors.items():
        for n in nei:
            G.add_edge(anchor_idx, n)
    
    
    '''
    print("Nodes of graph: ")
    print(G.nodes())
    print("Edges of graph: ")
    print(G.edges())
    '''
    if IF_WCC:
        components = nx.weakly_connected_components(G)
    else:
        components = nx.strongly_connected_components(G)
        
    cc = []
    for nodes in components:
        cc.append(list(nodes))

    

    return G, cc

def show_cc(pred_cc):
    for com in pred_cc:
        for i in range(len(com)):
            plt.subplot(1, len(com), i+1)
            plt.imshow(img[com[i]])

        plt.show()
        
        
def give_phrase_baseline(loc_array, text_content, pred_cc):
    pred_names = []
    for com in pred_cc:
        # sort according to x and y 
        x_list, y_list = [],[]
        for i in range(len(com)):
            x,y = loc_array[i]
            x_list.append(x)
            y_list.append(y)
            
            
        x_indices = np.argsort(x_list)
        y_indices = np.argsort(y_list)
        
        #print x_indices
        #print y_indices
        #print np.array(x_list)[x_indices]
        
        cur_name = []
        for i in range(len(com)):
            cur_name.append(text_content[com[x_indices[i]]])
        cur_name = " ".join(cur_name)
        pred_names.append(cur_name)
    
    return pred_names


def give_phrase_indices_baseline(loc_array, pred_cc):
    pred_names = []
    for com in pred_cc:
        # sort according to x and y 
        x_list, y_list = [],[]
        for i in range(len(com)):
            x,y = loc_array[i]
            x_list.append(x)
            y_list.append(y)
            
            
        x_indices = np.argsort(x_list)
        y_indices = np.argsort(y_list)
        
        cur_name = []
        for i in range(len(com)):
            #print text_content[com[x_indices[i]]]
            cur_name.append(com[x_indices[i]])
        pred_names.append(cur_name)
    
    return pred_names

def main(args):
    
    
    input_pred_dir = args.input_pred_dir
    vision_api_result_dir = args.vision_api_result_dir
    IF_WCC = args.IF_WCC # compute weakly connected component or strongly connected component
    output_phrases_dir = args.output_phrases_dir
    
    if not os.path.isdir(output_phrases_dir):
        os.makedirs(output_phrases_dir)

    map_names_list = [os.path.basename(a)[11:-4] for a in os.listdir(input_pred_dir)] # remove 'step2_pred_' prefix
    for map_name in map_names_list:
        pred_path = input_pred_dir + 'step2_pred_' + map_name + '.pkl'
        #gt_path = '/home/zekunl/preprocess_map/neighbors_dict_v3/neighbors_' + map_name + '.pkl'
        #text_path = '/home/zekunl/preprocess_map/text_dict_v3/text_' + map_name + '.pkl'
        #loc_path = '/home/zekunl/preprocess_map/loc_dict_v3/loc_' + map_name + '.pkl' 
        pred_text_content_path = vision_api_result_dir + '/word_list/'
        pred_loc_array = vision_api_result_dir + '/word_coords_list/'
        phrase_path = '/home/zekunl/preprocess_map/location_names/phrase_' + map_name + '.pkl' 
        output_path = output_phrases_dir  + map_name + '.pkl'

        print 'map_name', map_name 
        
        
        with open(pred_path, 'r') as f:
            predictions = pickle.load(f)

        #with open(gt_path, 'r') as f:
        #    ground_truths = pickle.load(f)

        #with open(text_path, 'r') as f:
        #    text_content =  pickle.load(f)

        #with open(loc_path, 'r') as f:
        #    loc_array =  pickle.load(f)    

        with open(phrase_path, 'r') as f:
            phrase_list = pickle.load(f)
            
        with open(pred_text_content_path + 'pkl2_'+ map_name + '.pkl', 'rb') as f:
            pred_text_content = pickle.load(f)

        with open(pred_loc_array +'pkl2_'+ map_name + '.pkl', 'rb') as f:
            pred_bbox_array = pickle.load(f)
        assert len(pred_text_content) == len(pred_bbox_array)
        
        pred_loc_array = []
        for bbox in pred_bbox_array:
            cur_point = np.array(bbox)
            c_x,c_y = np.mean(cur_point, axis = 0)
            pred_loc_array.append([c_x,c_y])

        # generate gt phrase
        loc_strings = []
        for phrase in phrase_list:
            loc_str = " ".join( phrase)
            loc_strings.append(loc_str)

        print 'loc_strings',loc_strings


        # In[182]:

        print 'number of location names', len(loc_strings)

        pred_G, pred_cc = construct_graph_cc_from_neighbor_dict(predictions, IF_WCC)
        #gt_G, gt_cc = construct_graph_cc_from_neighbor_dict(ground_truths, IF_WCC)

        pred_names = give_phrase_baseline(pred_loc_array, pred_text_content,pred_cc)
        #pred_indices = give_phrase_indices_baseline(loc_array,pred_cc)

        print pred_names
        '''
        #intersection_sets = set(tuple(cc) for cc in pred_cc).intersection(set(tuple(cc) for cc in gt_cc))
        #print 'unordered set recall', 1. *len(intersection_sets)/len(gt_cc), 'numbers:', len(intersection_sets)


        # In[189]:

        #intersection_sets = set(tuple(cc) for cc in pred_indices).intersection(set(tuple(cc) for cc in gt_cc))
        #print 'phrase precision', 1. *len(intersection_sets)/len(pred_indices), 'numbers:', len(intersection_sets)
        #print 'phrase recall', 1. *len(intersection_sets)/len(gt_cc), 'numbers:', len(intersection_sets)




        # Statistics
        #max_len = max([len(pred_ind) for pred_ind in pred_indices]) # get the maximum length of the location string


        #correct_cnt = [] # finally should have length == max_len, keep the number of correct predictions for all lengths
        #gt_cnt = []
        
        
        gt_lens = [len(gcc) for gcc in gt_cc] # length of each string in GT
        correct_lens = [len(inter_str) for inter_str in intersection_sets] # length of each string in prediction
        for le in range(1, max_len+1):
            pred_cur_cnt = len(np.where(np.array(correct_lens) == le)[0])
            gt_cur_cnt = len(np.where(np.array(gt_lens) == le)[0])
            correct_cnt.append(pred_cur_cnt) 
            gt_cnt.append(gt_cur_cnt)

        #print correct_cnt
        #:qprint gt_cnt

        for i in range(0, len(correct_cnt) ):
            print 'length == ' + str(i+1)
            if gt_cnt[i] == 0:
                print '# gt = 0'
                break
            
            print '# correct pred / # gt = ', str(correct_cnt[i]) + ' / '+ str(gt_cnt[i]) +' = ' + str(1. * correct_cnt[i]/gt_cnt[i])


        # In[137]:

        # statistic about missing and adding
        cnt_missing = 0
        cnt_adding = 0
        cnt_len_missing = [0 for i in range(10)]
        cnt_len_adding =[0 for i in range(10)]
        wrong_pred_names = list(set(tuple(cc) for cc in pred_indices) - intersection_sets)
        print 'wrong pred names', wrong_pred_names
        for wrong_cc in wrong_pred_names:
            for gcc in set(tuple(cc) for cc in gt_cc) :
                if set(wrong_cc).issubset(set(gcc)):
                    cnt_missing += 1
                    len_missing = len(gcc) - len(wrong_cc)
                    cnt_len_missing[len_missing-1] += 1
                    print " ".join([text_content[c] for c in wrong_cc]), wrong_cc, '\tmissing '+ str(len_missing) + ' word in\t', " ".join([text_content[c] for c in gcc]),gcc
                    continue
                if set(gcc).issubset( set(wrong_cc)):
                    cnt_adding += 1
                    len_adding = len(wrong_cc) - len(gcc)
                    cnt_len_adding[len_adding -1]+= 1
                    print " ".join([text_content[c] for c in wrong_cc]), wrong_cc, '\tadding ' + str(len_adding) + ' word in\t', " ".join([text_content[c] for c in gcc]), gcc
                    continue

        print 'cnt_missing',cnt_missing
        print 'cnt_adding', cnt_adding
        print 'cnt_len_missing', cnt_len_missing
        print 'cnt_len_adding', cnt_len_adding

                
        '''

        inter_string = set(pred_names).intersection(set(loc_strings))
        print 'inter string',inter_string


        # In[125]:

        print 'GT distinct # location names', len(set(loc_strings))
        print 'Pred distinct # location names', len(set(pred_names))
        print 'Pred correct # location names', len(inter_string)
        print 'Precision in distinct', 1. * len(inter_string) / len(set(pred_names))
        print 'Recall in distinct', 1. * len(inter_string) / len(set(loc_strings))
        
        
        
        
        print '\n\n\n'
        with open(output_path, 'wb') as f:
            pickle.dump(pred_names, f)
        
        

if __name__ == '__main__':
    args = parse_cmdline_args()
    print (args)
    main(args)
    
