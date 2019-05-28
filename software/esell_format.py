# Generate eSELL sparse matrix and compare with CSR, CSC, COO representation.

from scipy import sparse
import numpy as np
import sys


def esell_construct(mat_in, idx_encodebook, chk_h, n_chk, chk_w):
    '''
    :param mat_in: input matrix
    :param idx_encodebook: encodebook for col_idx encoding
    :param chk_h: chunk size (CHK_h), note: val_c should be an integer multiple of 4, to avoid the padding overhead
    :param n_chk: ration of BLK_h/CHK_h
    :param chk_w: CHK_W/BLK_W (should be an integer multiple of 4)
    :return: mat_esell, mat_permut, mat_chkw, mat_col_idx_encode
    '''
    # obtain the matrix size
    (row, col) = mat_in.shape

    mat_esell = np.zeros([row, col], dtype=mat_in.dtype)

    # sorting range size (BLK_h, sigma), note: should be an integer multiple of chunk height
    blk_h = n_chk * chk_h

    # row permutation index
    mat_permut = np.zeros([row, int(col / chk_w)], dtype='int16')
    # nnz of each block row, (after permutated)
    mat_nnz_permut = np.zeros([row, int(col / chk_w)], dtype='int16')
    # col index
    mat_col_idx = np.zeros([row, col], dtype='int16')
    mat_col_idx_permut = np.zeros([row, col], dtype='int16')
    # col index encode
    mat_col_idx_encode = np.zeros([row, int(col / chk_w)], dtype='int16')

    # chk_w
    mat_chkw = np.zeros([int(row / chk_h), int(col / chk_w)], dtype='int16')

    # mat slicing and sorting
    for idx_cw in range(0, int(col / chk_w)):
        for idx_blk in range(0, int(row / blk_h)):
            mat_slice = mat_in[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw * chk_w:(idx_cw + 1) * chk_w]
            # count and sort nnz
            vec_nnz = np.zeros(blk_h, dtype='int16')
            # iterate all line-elements of a block
            for t_blk in range(0, blk_h):
                nnz = np.count_nonzero(mat_slice[t_blk, :])
                vec_nnz[t_blk] = nnz
                # fill in the col idx
                if (nnz != 0):
                    mat_col_idx[idx_blk * blk_h + t_blk, idx_cw * chk_w:idx_cw * chk_w + nnz] = np.nonzero(mat_slice[t_blk, :])[0]
                    mat_esell[idx_blk * blk_h + t_blk, idx_cw * chk_w:idx_cw * chk_w + nnz] = mat_slice[t_blk, np.nonzero(mat_slice[t_blk, :])]

            permut_seq = np.argsort(-vec_nnz)
            mat_col_idx_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw * chk_w:(idx_cw + 1) * chk_w] = mat_col_idx[idx_blk * blk_h + permut_seq, idx_cw * chk_w:(idx_cw + 1) * chk_w]
            mat_esell[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw * chk_w:(idx_cw + 1) * chk_w] = mat_esell[idx_blk * blk_h + permut_seq, idx_cw * chk_w:(idx_cw + 1) * chk_w]
            mat_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw] = permut_seq

            # fill in the nnz matrix
            mat_nnz_permut[idx_blk * blk_h:(idx_blk + 1) * blk_h, idx_cw] = vec_nnz[permut_seq]
            for t_blk in range(0, blk_h):
                nnz = vec_nnz[permut_seq[t_blk]]
                if nnz == 0:
                    mat_col_idx_encode[idx_blk * blk_h + t_blk, idx_cw] = idx_encodebook[nnz]
                else:
                    col_idx_vld = tuple(mat_col_idx_permut[idx_blk * blk_h + t_blk, idx_cw * chk_w: idx_cw*chk_w+nnz])
                    mat_col_idx_encode[idx_blk * blk_h + t_blk, idx_cw] = idx_encodebook[nnz][col_idx_vld]

            # slicing the Chunk, be excuted in each BLK (idx_blk)
            for idx_nchk in range(0, n_chk):
                # STEP3: obtain CHK_w
                mat_chkw[idx_blk * n_chk + idx_nchk, idx_cw] = mat_nnz_permut[idx_blk * blk_h + idx_nchk * chk_h, idx_cw]

    print("Finish eSELL format construction.")
    return mat_esell, mat_permut, mat_chkw, mat_col_idx_encode


def esell_computation(mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, chk_h, n_chk, chk_w, idx_decodebook, vec_x):
    '''
    :param mat_esell:
    :param mat_permut:
    :param mat_chkw:
    :param mat_col_idx_encode:
    :param chk_h:
    :param n_chk:
    :param chk_w:
    :param idx_decodebook:
    :param vec_x:
    :return:
    '''

    (row, col) = mat_esell.shape
    blk_h = n_chk * chk_h

    # result vector
    vec_y = np.zeros(row)

    for idx_cw in range(0, int(col / chk_w)):
        for idx_blk in range(0, int(row / blk_h)):
            # obtain the permutation table
            permut_seq = mat_permut[idx_blk*blk_h:(idx_blk+1)*blk_h, idx_cw]
            for idx_nchk in range(0, n_chk):

                for jj_chkh in range(0, chk_h):
                    # obtain the col_idx_code & decode
                    col_idx_code = mat_col_idx_encode[idx_blk * blk_h + idx_nchk*chk_h + jj_chkh, idx_cw]
                    col_idx = idx_decodebook[:, col_idx_code]
                    # number of iteration of ii_chkw depends on the records in mat_chkw
                    for ii_chkw in range(0, mat_chkw[idx_blk*n_chk+idx_nchk, idx_cw]):
                        elem_x = vec_x[idx_cw*chk_w+col_idx[ii_chkw]]
                        multi_res = mat_esell[idx_blk*blk_h+idx_nchk*chk_h+jj_chkh, idx_cw*chk_w+ii_chkw] * elem_x
                        vec_y[idx_blk * blk_h + permut_seq[idx_nchk * chk_h + jj_chkh]] = vec_y[idx_blk*blk_h+permut_seq[idx_nchk*chk_h+jj_chkh]] + multi_res

    print("Finish computation with eSELL format.")
    return vec_y


def dump_weight(f, mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, w_name, ROW, COL, chk_h, n_chk, chk_w):
    # write values
    f.write('uint64_t {:s}[]={{'.format(w_name))
    # output hex header
    addr_offset_rocc = []
    for idx_blkcol in range(0, int(COL/chk_w)):
        cnt_rocc = 0
        for idx_blk in range(0, int(ROW/chk_h/n_chk)):
            # obtain the block head
            blk_head = 0
            for idx_chk in range(0,n_chk):
                w_chk = mat_chkw[idx_blk*n_chk+idx_chk, idx_blkcol]
                row_idx = 0
                col_eidx = 0
                for ii in range(0,chk_h):
                    row_idx = (mat_permut[(idx_blk*n_chk+idx_chk)*chk_h+ii, idx_blkcol] << (ii*3)) | row_idx
                    col_eidx = (mat_col_idx_encode[(idx_blk*n_chk+idx_chk)*chk_h+ii, idx_blkcol] << (ii*3)) | col_eidx
                chk_head = int(w_chk) | (col_eidx<<3) | (row_idx << 15)
                blk_head = blk_head | (chk_head << (32 * idx_chk))
            # output blk_head
            f.write("{:d},".format(blk_head))
            cnt_rocc = cnt_rocc + 1
            #
            for idx_chk in range(0, n_chk):
                w_chk = mat_chkw[idx_blk * n_chk + idx_chk, idx_blkcol]
                for tt in range(0, w_chk):
                    # one rocc word (uint64)
                    elem_uint64 = 0
                    for ii in range(0, chk_h):
                        elem_uint16 = mat_esell[(idx_blk*n_chk+idx_chk)*chk_h+ii, idx_blkcol*chk_w+tt].view('H')
                        elem_uint64 = elem_uint64 | (elem_uint16 << (ii*16))
                    # output elem_uint64
                    f.write("{:d},".format(elem_uint64))
                    cnt_rocc = cnt_rocc + 1
        # append the cnt_rocc to the list
        addr_offset_rocc.append(cnt_rocc)
    f.write("};\n")

    # dump the block column rocc address offset
    f.write('uint64_t {:s}_offset[{:d}]={{'.format(w_name, int(COL/chk_w/4)+1))
    tmp = 0
    for ii in range(0, int(COL/chk_w)):
        tmp = tmp | addr_offset_rocc[ii] << (16*(ii%4))
        if (ii%4)==3 or ii==(COL/chk_w-1):
            f.write("{:d},".format(tmp))
            tmp = 0
    f.write("};\n")


def gen_codebook():
    # define the idx_encodebook (e.g., chk_w=4)
    idx_encodebook = []
    idx_encodebook.append(0)

    tmp_mat = np.zeros([4], dtype='int16')
    tmp_mat[:] = [0,4,6,7]
    idx_encodebook.append(tmp_mat)

    tmp_mat = np.zeros([4,4], dtype='int16')
    tmp_mat[0,1] = 0; tmp_mat[0,2] = 2; tmp_mat[0,3] = 3; tmp_mat[1,2] = 4; tmp_mat[1,3] = 5; tmp_mat[2,3] = 6
    idx_encodebook.append(tmp_mat)

    tmp_mat = np.zeros([4,4,4], dtype='int16')
    tmp_mat[0,1,2] = 0; tmp_mat[0,1,3] = 1; tmp_mat[0,2,3] = 2; tmp_mat[1,2,3] = 4
    idx_encodebook.append(tmp_mat)

    tmp_mat = np.zeros([4,4,4,4], dtype='int16')
    tmp_mat[0,1,2,3] = 0
    idx_encodebook.append(tmp_mat)

    # define the idx_decodebook (chk_w=4)
    idx_decodebook = np.zeros([4, 8], dtype='int16')
    idx_decodebook[0][[4,5]] = 1
    idx_decodebook[0][6] = 2
    idx_decodebook[0][7] = 3
    idx_decodebook[1][[0,1]] = 1
    idx_decodebook[1][[2,4]] = 2
    idx_decodebook[1][[3,5,6]] = 3
    idx_decodebook[2][0] = 2
    idx_decodebook[2][[1,2,4]] = 3
    idx_decodebook[3][[0]] = 3
    return idx_encodebook, idx_decodebook


if __name__ == '__main__':
    # variable from input arguments
    argv = sys.argv
    ROW = int(argv[1])
    COL = int(argv[2])
    DENSITY = float(argv[3])

    # generate the sparse matrix in COO format
    mat_coo = sparse.rand(ROW, COL, density=DENSITY, format='coo', random_state=666)
    mat_coo = mat_coo.astype('float16')

    mat_csr = mat_coo.tocsr()
    mat_csc = mat_coo.tocsc()
    mat_array = mat_coo.astype('float').toarray()

    # generate dense vector
    np.random.seed(666)
    vec_float = np.random.rand(COL)
    # SpMV reference results
    res = np.dot(mat_array.astype('float16'), vec_float)


    # eSELL format parameters
    (chk_h, n_chk, chk_w) = (4,2,4)

    idx_encodebook, idx_decodebook = gen_codebook()

    # trans to eSELL-C-\sigma format
    mat_esell, mat_permut, mat_chkw, mat_col_idx_encode = esell_construct(mat_array.astype('float16'), idx_encodebook, chk_h, n_chk, chk_w)
    vec_y = esell_computation(mat_esell, mat_permut, mat_chkw, mat_col_idx_encode, chk_h, n_chk, chk_w, idx_decodebook, vec_float)

    # verification
    diff = vec_y - res
    assert(diff.all()==0)
    print("Success.")
