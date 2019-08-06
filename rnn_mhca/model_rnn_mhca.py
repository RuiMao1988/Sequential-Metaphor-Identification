import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch
from attention import MultiHeadContextAttention
import numpy as np
from torch.autograd import Variable


class RNNSequenceModel(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.5, dropout2=0, dropout3=0.2):

        super(RNNSequenceModel, self).__init__()

        self.rnn = nn.LSTM(input_size=embedding_dim , hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout2, batch_first=True, bidirectional=bidir)

        direc = 2 if bidir else 1

        self.output_to_label = nn.Linear(hidden_size * direc * 2, num_classes)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)
        self.dropout_on_input_to_linear_layer = nn.Dropout(dropout3)

        self.multiheadcontextattention = MultiHeadContextAttention(input_depth=hidden_size, 
                 total_key_depth=64, total_value_depth=64, output_depth=hidden_size, 
                 num_heads=16, bias_mask=None, dropout=0.0)

    def get_context(self, input_l2r, input_r2l, window = 3):
    
        if window == False:
            
            batch_size = input_l2r.size(0)
            input_seq_len = input_l2r.size(1) 
            input_l2r = torch.cat([torch.zeros(input_l2r.size(0), 1, input_l2r.size(2)).type(torch.FloatTensor).cuda(), input_l2r], 1)
            input_r2l = torch.cat([input_r2l, torch.zeros(input_l2r.size(0), 1, input_l2r.size(2)).type(torch.FloatTensor).cuda()], 1)

            context_l2r = input_l2r.unsqueeze(1).expand(batch_size, input_seq_len, input_l2r.size(1), 
                                                    input_l2r.size(2)).contiguous().view(batch_size*input_seq_len,
                                                                                         input_l2r.size(1),input_l2r.size(2))
            context_r2l = input_r2l.unsqueeze(1).expand(batch_size, input_seq_len, input_r2l.size(1), 
                                                    input_r2l.size(2)).contiguous().view(batch_size*input_seq_len,
                                                                                         input_r2l.size(1),input_r2l.size(2))
            mask_eye = np.eye(input_seq_len, dtype=int)
            zero_list = np.zeros((input_seq_len), dtype=int).tolist()
            mask_tensor_l2r_list = []
            mask_tensor_r2l_list = []
            
            # forming masks based on the sequence length
            for i in range(input_seq_len):
                mask_eye_l2r_list = mask_eye.tolist()
                mask_eye_l2r_list.insert(i+1, zero_list)
                mask_eye_l2r_array = np.array(mask_eye_l2r_list)
                mask_eye_l2r_array[i+1:, :] = 0
                mask_tensor_l2r_list.append(mask_eye_l2r_array.tolist())
                
                mask_eye_r2l_list = mask_eye.tolist()
                mask_eye_r2l_list.insert(i, zero_list)
                mask_eye_r2l_array = np.array(mask_eye_r2l_list)
                mask_eye_r2l_array[:i+1, :] = 0
                mask_tensor_r2l_list.append(mask_eye_r2l_array.tolist())
     
                
            mask_tensor_l2r = torch.FloatTensor(mask_tensor_l2r_list)
            mask_tensor_r2l = torch.FloatTensor(mask_tensor_r2l_list)
            
            mask_l2r = mask_tensor_l2r.unsqueeze(0).expand(batch_size, mask_tensor_l2r.size(0), 
                                                           mask_tensor_l2r.size(1), mask_tensor_l2r.size(-1))
            mask_r2l = mask_tensor_r2l.unsqueeze(0).expand(batch_size, mask_tensor_r2l.size(0), 
                                                           mask_tensor_r2l.size(1), mask_tensor_r2l.size(-1))
            
        else:
        
            batch_size = input_l2r.size(0)
            input_seq_len = input_l2r.size(1)
            pad_window = torch.zeros(input_l2r.size(0), window, input_l2r.size(2)).cuda()
            input_l2r = torch.cat([pad_window, input_l2r], 1)
            input_r2l = torch.cat([input_r2l, pad_window], 1)
            context_l2r = input_l2r.unsqueeze(1).expand(batch_size, input_seq_len, input_l2r.size(1), 
                                                    input_l2r.size(2)).contiguous().view(batch_size*input_seq_len,
                                                                                         input_l2r.size(1),input_l2r.size(2))
            context_r2l = input_r2l.unsqueeze(1).expand(batch_size, input_seq_len, input_r2l.size(1), 
                                                    input_r2l.size(2)).contiguous().view(batch_size*input_seq_len,
                                                                                         input_r2l.size(1),input_r2l.size(2))
            mask_eye = torch.from_numpy(np.eye(window, dtype=int)).type(torch.FloatTensor)

            mask_tensor_l2r = torch.cat([mask_eye, torch.zeros(context_l2r.size(1)-mask_eye.size(0),
                                                               mask_eye.size(1)).type(torch.FloatTensor)], 0)
            mask_tensor_r2l = torch.cat([torch.cat([torch.zeros(1, mask_eye.size(-1)).type(torch.FloatTensor), 
                                                    mask_eye], 0), torch.zeros(context_l2r.size(1)-mask_eye.size(0)-1,
                                                     mask_eye.size(1)).type(torch.FloatTensor)], 0)

            # forming masks based on the window size
            for i in range(1, context_l2r.size(1)-window):
                mask_tensor_l2r = torch.cat([mask_tensor_l2r, 
                                             torch.zeros(i, mask_eye.size(1)).type(torch.FloatTensor)], 0)
                mask_tensor_l2r = torch.cat([mask_tensor_l2r,
                                            mask_eye], 0)
                mask_tensor_l2r = torch.cat([mask_tensor_l2r,
                                            torch.zeros(context_l2r.size(1)-i-mask_eye.size(0), 
                                                        mask_eye.size(1)).type(torch.FloatTensor)], 0)

                mask_tensor_r2l = torch.cat([mask_tensor_r2l, 
                                             torch.zeros(i+1, mask_eye.size(1)).type(torch.FloatTensor)], 0)
                mask_tensor_r2l = torch.cat([mask_tensor_r2l,
                                            mask_eye], 0)
                mask_tensor_r2l = torch.cat([mask_tensor_r2l,
                                            torch.zeros(context_r2l.size(1)-i-1-mask_eye.size(0), 
                                                        mask_eye.size(1)).type(torch.FloatTensor)], 0)
            mask_l2r = mask_tensor_l2r.unsqueeze(0).expand(batch_size, mask_tensor_l2r.size(0), mask_tensor_l2r.size(1))
            mask_r2l = mask_tensor_r2l.unsqueeze(0).expand(batch_size, mask_tensor_r2l.size(0), mask_tensor_r2l.size(1))

        mask_l2r = Variable(mask_l2r.contiguous().view(context_l2r.size(0), context_l2r.size(1), -1), 
                            requires_grad = False).cuda()
        mask_r2l = Variable(mask_r2l.contiguous().view(context_r2l.size(0), context_r2l.size(1), -1), 
                            requires_grad = False).cuda()

        maskted_context_l2r = torch.bmm(context_l2r.permute(0,2,1),mask_l2r).permute(0,2,1)
        maskted_context_r2l = torch.bmm(context_r2l.permute(0,2,1),mask_r2l).permute(0,2,1)
        
        return maskted_context_l2r, maskted_context_r2l

    def get_query(self, input):
        query = input.unsqueeze(2).view(input.size(0)*input.size(1), 1, input.size(2))
        return query

    def forward(self, inputs, lengths):

        embedded_input = self.dropout_on_input_to_LSTM(inputs)

        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)

        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)

        packed_sorted_output, _ = self.rnn(packed_input)

        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)

        output = sorted_output[input_unsort_indices]

        att_vec_dim = int(output.size(2)/2)
        query_l2r = self.get_query(output[:, :, :att_vec_dim])
        query_r2l = self.get_query(output[:, :, att_vec_dim:])
        context_l2r, context_r2l = self.get_context(output[:, :, :att_vec_dim], output[:, :, att_vec_dim:], window = 3)

        att_l2r = self.multiheadcontextattention(query_l2r, context_l2r, context_l2r).view(output.size(0), output.size(1), -1)
        att_r2l = self.multiheadcontextattention(query_r2l, context_r2l, context_r2l).view(output.size(0), output.size(1), -1)

        att = torch.cat([att_l2r, att_r2l], -1)

        output_cat = torch.cat([output, att], -1)

        input_encoding = self.dropout_on_input_to_linear_layer(output_cat)

        unnormalized_output = self.output_to_label(input_encoding)

        output_distribution = F.log_softmax(unnormalized_output, dim=-1)
        return output_distribution
