# LSTM functional simulation and bit-packaging of eSELL format for ROCC interface

from scipy import sparse
import numpy as np
import sys
import esell_format as esell


def Sigmod(input):
    return 1/(1+np.exp(-input))


if __name__ == '__main__':
    x_size = 1500;
    h_size = 1500;
    # time step in one sequence processing
    ts_size = 35;


    np.random.seed(666)

    # eSELL format parameters
    (chk_h, n_chk, chk_w) = (4, 2, 4)

    idx_encodebook, idx_decodebook = esell.gen_codebook()

    # # generate the sparse matrix W
    # w_mat_coo = sparse.rand(h_size*4, x_size, density=0.6, format='coo', random_state=666)
    # w_mat_coo = w_mat_coo.astype('float16')
    # # dense array
    # w_mat_array = w_mat_coo.astype('float').toarray()*0.005-0.0025

    load_data = np.load('wiki-l2.npz')

    w_mat_array = load_data['w']

    w_mat_esell, w_mat_permut, w_mat_chkw, w_mat_col_idx_encode = esell.esell_construct(w_mat_array.astype('float16'), idx_encodebook, chk_h, n_chk, chk_w)


    # # generate the sparse matrix U
    # u_mat_coo = sparse.rand(h_size*4, h_size, density=0.6, format='coo', random_state=777)
    # u_mat_coo = u_mat_coo.astype('float16')
    # # dense array
    # u_mat_array = u_mat_coo.astype('float').toarray()*0.005-0.0025

    u_mat_array = load_data['u']

    u_mat_esell, u_mat_permut, u_mat_chkw, u_mat_col_idx_encode = esell.esell_construct(u_mat_array.astype('float16'), idx_encodebook, chk_h, n_chk, chk_w)

    # generate bias(vec bias)
    # vec_bias = np.random.rand(h_size*4)

    vec_bias = load_data['b']

    # generate sequence (vec x)
    # vec_x = np.random.rand(x_size*ts_size)


    # simulator the computation

    # h_t = np.zeros(h_size)
    # c_t = np.zeros(h_size)
    #
    # for ii in range(0, ts_size):
    #     vec_wx = np.dot(w_mat_array.astype('float16'), vec_x[ii*x_size:(ii+1)*x_size])
    #     vec_sum = vec_wx + vec_bias + np.dot(u_mat_array.astype('float16'), h_t)
    #     # vec_sum = res + bias
    #     f_t = Sigmod(vec_sum[0:h_size])
    #     i_t = Sigmod(vec_sum[h_size:2 * h_size])
    #     c_t = f_t * c_t + i_t * np.tanh(vec_sum[2 * h_size:3 * h_size])
    #     o_t = Sigmod(vec_sum[3 * h_size:4 * h_size])
    #     h_t = o_t * np.tanh(c_t)
    # print(h_t)
    # print("End of simulation.")

    # dump file for RISC-V ELSTM simulator
    f = open("wiki_l2.h", "w")
    esell.dump_weight(f, w_mat_esell, w_mat_permut, w_mat_chkw, w_mat_col_idx_encode, 'w', h_size*4, x_size, 4,2,4)
    esell.dump_weight(f, u_mat_esell, u_mat_permut, u_mat_chkw, u_mat_col_idx_encode, 'u', h_size*4, h_size, 4,2,4)

    # dump the vec bias
    f.write('uint64_t bias[{:d}]={{'.format(int(h_size * 4 / 4)))
    bias_float16 = vec_bias.astype('float16')
    tmp = 0
    for ii in range(0, h_size * 4):
        tmp = tmp | bias_float16[ii].view('H') << (16 * (ii % 4))
        if (ii % 4) == 3 or ii == (len(bias_float16) - 1):
            f.write("{:d},".format(tmp))
            tmp = 0
    f.write("};\n")

    load_data = np.load('wiki-h.npz')
    vec_x = load_data['x']
    f.write('float x[{:d}]={{'.format(len(vec_x)))
    # repeat multiple vector x
    for ii in range(0, len(vec_x)):
        f.write("{:f},".format(vec_x[ii]))
    f.write("};\n")

    f.close()