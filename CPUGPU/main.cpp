// Copyright (c) 2018 NSF Center for Space, High-performance, and Resilient Computing (SHREC)
// University of Pittsburgh. All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided
// that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#define USE_ONEAPI_FUNCTIONS 0

#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <fstream>
#include <stdint.h>
#include <arpa/inet.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#if USE_ONEAPI_FUNCTIONS
#include <mkl.h>
#include <oneapi/dpl/random>
#include <oneapi/mkl/blas.hpp>
#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/vm.hpp"
#endif



#define TIME (std::chrono::high_resolution_clock::now())
#define FOR(i,n) for(int i = 0; i < n; i++)
#define FORX(it,iterable) for(auto &it : iterable)

#define DATA_PATH PROJECT_PATH_CMAKE "/CPUGPU/data/"

using namespace cl::sycl;


using std::cout;

//typedefs
typedef std::vector<std::vector<float>> fmat;
typedef std::vector<float> fvec;
typedef std::vector<int> ivec;

typedef buffer<float,1> fbuf;
typedef buffer<int,1> ibuf;

typedef std::chrono::high_resolution_clock::time_point timep;

#define MTRANS oneapi::mkl::transpose::trans
#define MNOTRANS oneapi::mkl::transpose::nontrans

typedef struct{
    fmat data;
    ivec labels;
    int32_t numFeatures;
    int32_t numClasses;
} Data;

//prototypes
double elapsedTime(timep start, timep end);
fvec normalize(fvec input);
Data readData(const char *filename);
void trainAndTestOneShot();
void testInferenceBaseline();

//performs matrix mult C = A * trans(B)
//where d1 is rows of A, d2 is cols A or rows B, d3 is cols B
void mmult(fbuf &A, fbuf &B, fbuf &C, int d1, int d2, int d3);

//generate random basis matrix for conversion into hyper-space
//matrix mult basis with data_buf -> out_buf
//out_buf is data_buf vectors in hyperspace
void encode(fbuf &data_buf, fbuf &out_buf, int ndata);

//X = classes * trans(data)
//classes: 1 class per row, dim cols
//data: ndata per row, dim cols... trans(data): dim rows, ndata cols    
//X: nclasses rows, ndata cols
//rows of X are now data points dotted with classes
//X(row,col) = data[row] closeness to class[col]
//guesses = argmax of each row
//correct = number of matches between guesses and labels
//ALSO for each data point assigned incorrectly update:
//adjust correct class add mislabeled data * step size
//adjust incorrect class subtract mislabeled data * step size
void fit(fbuf &data, ivec &labels, int &correct);
void fit2(fbuf &data, ivec &labels, int &correct);
//slightly different, bundles classes instead
void fitOneShot(fbuf &data, ivec &labels, int &correct);

//same as fit, just dont update
void testNN(fbuf &data, ivec &labels, int &correct);
void testNN2(fbuf &data, ivec &labels, int &correct);

//ranks 
ivec rankDims();

void updateClassesAndBasis(ivec &dim_ranks);

//global variables
static int32_t nfeatures, nclasses, ndims, work_group_size;
static fvec basisv, classesv;
static fbuf *basis_bufp = nullptr, *classes_bufp = nullptr;
static queue *q = nullptr;
static fvec accuracies, inference_times, train_times, runtimes;

timep tstart, tend;

device d_selector(default_selector_v);

static auto e_handler = [](exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};

double elapsedTime(timep start, timep end) {
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
  return time_span.count();
}

fvec normalize(fvec input) {
	float inner = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
	float norm = std::sqrt(inner);
	for(int i = 0; i < input.size(); i++) {
		input[i] = input[i] / norm;
	}
	return input;
}

fvec m2v(fmat in){
	fvec out(in.size() * in[0].size(), 0.0);
	for(int i = 0; i < in.size(); i++) {
		for(int j = 0; j < in[0].size(); j++) {
			out[i * in[0].size() + j] = in[i][j];
		}
	}
    return out;
}
#if USE_ONEAPI_FUNCTIONS
void mmult(fbuf &A, fbuf &B, fbuf &C, int d1, int d2, int d3){
    oneapi::mkl::blas::row_major::gemm( *q,
                                        MNOTRANS,
                                        MTRANS,
                                        d1,
                                        d2,
                                        d3,
                                        1.0,
                                        A,
                                        d3,
                                        B,
                                        d3,
                                        0,
                                        C,
                                        d2);
}
#else
void mmult(fbuf &Data_b, fbuf &Basis_b, fbuf &out_buf_b, int ndata, int ndims, int nfeat){
    range<2> r(ndata,ndims);
    q->submit([&](auto &h){
        accessor Data_b_a(Data_b, h, read_only);
        accessor Basis_b_a(Basis_b, h, read_only);
        accessor out_buf_a(out_buf_b, h, write_only, no_init);
        h.parallel_for(r, [=](auto lr){
            int dt = lr[0];
            int dim = lr[1];
            out_buf_a[dt*ndims+dim] = 0.0;
            for(int f = 0; f < nfeat; f++) {
                out_buf_a[dt*ndims+dim] += Data_b_a[dt*nfeat+f] * Basis_b_a[dim*nfeat+f];
            }
        });
    });
}
#endif


Data readData(char *filename) {
    std::ifstream testFile(filename, std::ifstream::binary);
    char *holder = (char *)malloc(4 * sizeof(char));
    testFile.read(holder, 4 * sizeof(char));
    int32_t numFeatures;
    memcpy(&numFeatures, &holder[0], sizeof(numFeatures));
    testFile.read(holder, 4 * sizeof(char));
    int32_t numClasses;
    memcpy(&numClasses, &holder[0], sizeof(numClasses));
    fmat testData;
    ivec testLabels;
    while(testFile.good()) {
        fvec vect(numFeatures, 0.0);
        bool zero = true;
        for(int i = 0; i < numFeatures; i++) {
            testFile.read(holder, 4 * sizeof(char));
            float val;
            memcpy(&val, &holder[0], sizeof(val));
            vect[i] = val;
            if(val != 0.0) {
                zero = false;
            }
        } 
        testFile.read(holder, 4 * sizeof(char));
        int label;
        memcpy(&label, &holder[0], sizeof(label));
        if(!zero) {
            vect = normalize(vect);
            testData.push_back(vect);
            testLabels.push_back(label);
        }
    }
    free(holder);
    testFile.close();
    Data ret;
    ret.data = testData;
    ret.labels = testLabels;
    ret.numClasses = numClasses;
    ret.numFeatures = numFeatures;
    return ret;
}

void encode(fbuf &data_buf, fbuf &out_buf, int ndata){
    mmult(data_buf, *basis_bufp, out_buf, ndata, ndims, nfeatures);
    q->wait();
#if USE_ONEAPI_FUNCTIONS
    oneapi::mkl::vm::cos(*q, ndata*ndims, out_buf, out_buf);
#else
    auto r = out_buf.get_range();
    q->submit([&](auto &h){
        accessor out_a(out_buf, h, read_write);
        h.parallel_for(r, [=](auto i){
            out_a[i] = cos(out_a[i]);
        });
    });
#endif
    q->wait();
}

int cpuFit(fbuf &dataBuf, ivec &labels, fvec &classes) {
    host_accessor data(dataBuf);
    int correct = 0;
    ivec guesses(labels.size(), 0);
    for(int i = 0; i < labels.size(); i++) {
        float max = 0.0f;
        int guess = 0;
        for(int j = 0; j < nclasses; j++) {
            float sum = 0.0f;
            for(int k = 0; k < ndims; k++) {
                sum += classes[j * ndims + k] * data[i * ndims + k];
            }
            if(sum > max) {
                max = sum;
                guess = j;
            }
        }
        guesses[i] = guess;
        if(guess == labels[i]) {
            correct ++;
        }
        else {
            for(int j = 0; j < ndims; j++) { 
                classes[guess * ndims + j] -= 0.037 * data[i * ndims + j];
                classes[labels[i] * ndims + j] += 0.037 * data[i * ndims + j];
            }
        }
    }
    return correct;
}

void fit(fbuf &data, ivec &labels, int &correct){
    ivec indexes(labels.size());
    FOR(i, labels.size()) indexes[i] = i;
    srand(time(0));
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indexes.begin(), indexes.end(),g);

    int group_size = 1;
    int ndata = labels.size();

    ivec guesses(ndata);
    ibuf guesses_buf(guesses);
    ibuf label_buf(labels);
    ibuf indexes_buf(indexes);
    fvec classes_pre_norm(nclasses * ndims, 0.0);
    fbuf classes_pre_norm_buf(classes_pre_norm);
    q->submit([&](auto &h){
        accessor label_a(label_buf, h, read_only);
        accessor data_a(data, h, read_only);
        accessor class_a(*classes_bufp, h, read_only);
        accessor indexes_a(indexes_buf, h, read_only);
        accessor guesses_a(guesses_buf, h, write_only);
        
        int ndims_ = ndims;
        int nclasses_ = nclasses;
        int ndata_ = ndata;

        h.parallel_for(range(ndata_), [=](auto i){
            float max = -1.0;
            int guess = -1;
            int index = indexes_a[i[0]];
            FOR(j, nclasses_){
                float cum = 0;
                FOR(k, ndims_){
                    cum += data_a[index*ndims_+k] * class_a[j*ndims_+k];
                }
                if(cum > max){
                    max = cum;
                    guess = j;
                }
            }
            guesses_a[index] = guess;
        });
    });
    q->wait();
    q->submit([&](auto &h){
        accessor label_a(label_buf, h, read_only);
        accessor data_a(data, h, read_only);
        accessor indexes_a(indexes_buf, h, read_only);
        accessor guesses_a(guesses_buf, h, read_only);
        accessor class_a(*classes_bufp, h, read_write);
        
        int ndims_ = ndims;
        int nclasses_ = nclasses;
        int ndata_ = ndata;
        float step = 0.037;

        h.parallel_for(range(ndims_), [=](auto dim){
            FOR(i,ndata_){
                int index = indexes_a[i];
                if (label_a[index] != guesses_a[index]){
                    class_a[ndims_*guesses_a[index]+dim[0]] -= data_a[ndims_*index+dim[0]] * step;
                    class_a[ndims_*label_a[index]+dim[0]] += data_a[ndims_*index+dim[0]] * step;
                }
            }
        });
    });
    q->wait();
}

void fitOneShot(fbuf &data, ivec &labels, int &correct){
    int ndata = labels.size();

    ibuf label_buf(labels);
    fvec classes_pre_norm(nclasses * ndims, 0.0);
    fbuf classes_pre_norm_buf(classes_pre_norm);
    q->submit([&](auto &h){
        accessor data_a(data, h, read_only);
        accessor label_a(label_buf, h, read_only);
        accessor classes_pre_norm_a(classes_pre_norm_buf, h, read_write);

        int nclasses_ = nclasses;
        int ndata_ = ndata;
        int ndims_ = ndims;

        h.parallel_for(range(ndims_), [=](auto dim){
            FOR(c, nclasses_){
                FOR(dat, ndata_){
                    if (label_a[dat] == c){
                        classes_pre_norm_a[c * ndims_ + dim] += data_a[dat * ndims_ + dim];
                    }
                }
            }
        });
    });
    cout << "submit 1" << std::endl;
    q->wait();
    cout << "submit 1 done" << std::endl;
    q->submit([&](auto &h){
        accessor classes_pre_norm_a(classes_pre_norm_buf, h, read_only);
        accessor class_a(*classes_bufp, h, write_only);

        int nclasses_ = nclasses;
        int ndata_ = ndata;
        int ndims_ = ndims;

        h.parallel_for(range(nclasses_), [=](auto c){
            float sum = 0.0;
            FOR(i, ndims_){
                float val = classes_pre_norm_a[c * ndims_ + i];
                sum += val * val;
            }
            float invmag = (float)1.0 / sqrtf(sum);
            FOR(i, ndims_){
                class_a[c * ndims_ + i] = classes_pre_norm_a[c * ndims_ + i] * invmag;
            }
        });
    });
    cout << "submit 2" << std::endl;
    q->wait();
    cout << "submit 2 done" << std::endl;
}

double test_times[2];
#if 0
void testNN(fbuf &data, ivec &labels, int &correct){
    int ndata = labels.size();
    fvec intermed(ndata * nclasses, 0);
    fbuf intermed_buf(intermed);
    ibuf correct_buf(&correct, 1);
    ibuf label_buf(labels);
    auto t1 = TIME;
    mmult(data, *classes_bufp, intermed_buf, ndata, nclasses, ndims);
    q->wait();
    test_times[0] += elapsedTime(t1,TIME);
    int group_size = q->get_device().get_info<info::device::max_work_group_size>();
    int num_items = ndata;
    if (ndata % group_size != 0){
        num_items += group_size - (ndata % group_size);
    }
    t1 = TIME;

    int sums = 0;

    {
        host_accessor b(correct_buf,read_only);
        for(auto v:b)
            sums += v;
    }

    q->submit([&](auto &h){
        accessor intermed_a(intermed_buf, h, read_only);
        accessor label_a(label_buf, h, read_only);
        accessor correct_a(correct_buf, h, read_write);

        int nclasses_ = nclasses;
        int ndata_ = ndata;

        h.parallel_for(nd_range<1>(num_items, group_size), sums, [=](auto item, auto &sum_arg){
            size_t global_id = item.get_global_id(0);

            if(global_id < ndata_) {
                float max = -1.0;
                int guess = -1;
                for (int i = 0; i < ndata_; i++){
                    float temp = intermed_a[global_id * ndata_ + i];
                    if(temp > max) {
                        max = temp;
                        guess = i;
                    }
                }
                sum_arg += (int)(guess == label_a[global_id]);
            }
        });
    });
    q->wait();
    test_times[1] += elapsedTime(t1,TIME);
}
#endif

void testNN2(fbuf &data, ivec &labels, int &correct){
    int ndata = labels.size();

    ibuf label_buf(labels);
    ivec correct_vec(ndata, 0);
    ibuf correct_buf(correct_vec);

    buffer<int, 1> correct_buf_single(&correct, 1);
    q->submit([&](auto &h){
        accessor label_a(label_buf, h, read_only);
        accessor data_a(data, h, read_only);
        accessor class_a(*classes_bufp, h, read_only);
        accessor correct_a(correct_buf, h, read_write);
        
        int ndims_ = ndims;
        int nclasses_ = nclasses;
        int ndata_ = ndata;

        h.parallel_for(range(ndata_), [=](auto i){
            float max = -1.0;
            int guess = -1;
            int index = i[0];
            FOR(j, nclasses_){
                float cum = 0;
                FOR(k, ndims_){
                    cum += data_a[index*ndims_+k] * class_a[j*ndims_+k];
                }
                if(cum > max){
                    max = cum;
                    guess = j;
                }
            }
            if(guess == label_a[index]){
                correct_a[index] = 1;
            }
        });
    });
    q->wait();
    q->submit([&](auto &h){
        accessor correct_a(correct_buf, h, read_only);
        accessor correct_a_out(correct_buf_single, h, read_write);
        
        int ndims_ = ndims;
        int nclasses_ = nclasses;
        int ndata_ = ndata;

        h.single_task([=](){
            FOR(i, ndata) correct_a_out[0] += correct_a[i];
        });
    });
    q->wait();
}

ivec rankDims(){
    fvec variances(ndims);
    fbuf *var_buf = new fbuf(variances);
    ivec sorted_indexes(ndims);
    std::iota(sorted_indexes.begin(), sorted_indexes.end(), 0);
    ibuf sorted_indexes_buf(sorted_indexes);

    q->submit([&](auto &h){
        accessor class_a(*classes_bufp, h, read_only);
        accessor var_a(*var_buf, h, write_only);

        int ndims_ = ndims;
        int nclasses_ = nclasses;

        h.parallel_for(range(ndims_), [=](auto index){
            int d = index[0];
            float mean=(float)0, sum=(float)0;
            FOR(c, nclasses_){
                mean += class_a[c*ndims_+d];
            }
            mean *= (float)1 / (float)nclasses_;
            FOR(c, nclasses_){
                float temp = class_a[c*ndims_+d] - mean;
                sum += temp*temp;
            }
            var_a[d] = sum;
        });
    });
    q->wait();
    delete var_buf; var_buf = nullptr;
    //sort indeces in descending order
    std::sort(sorted_indexes.begin(), sorted_indexes.end(),
        [&](int i, int j){ return (variances[i] < variances[j]); });
    return sorted_indexes;
}
#if 0
void updateClassesAndBasis(ivec &dim_ranks){
    int dim_loss = dim_ranks.size();
    ibuf dim_ranks_buf(dim_ranks);
    
    srand (time(NULL));
    int seed = rand();
    q->submit([&](auto &h){
        accessor basis_a(*basis_bufp, h, read_write);
        accessor dim_ranks_a(dim_ranks_buf, h, read_only);

        int nfeatures_ = nfeatures;
        int ndims_ = ndims;

        h.parallel_for(range(nfeatures_, dim_loss), [=](auto index){
			std::uint64_t offset = index.get_linear_id();
			oneapi::dpl::minstd_rand engine(seed, offset);
			oneapi::dpl::normal_distribution<float> distr;
			basis_a[index[0]*ndims_ + dim_ranks_a[index[1]]] = distr(engine);
        });
    });
    q->submit([&](auto &h){
        accessor class_a(*classes_bufp, h, read_write);
        accessor dim_ranks_a(dim_ranks_buf, h, read_only);

        int nclasses_ = nclasses;
        int ndims_ = ndims;

        h.parallel_for(range(nclasses_, dim_loss), [=](auto index){
            class_a[index[0]*ndims_ + dim_ranks_a[index[1]]] = (float)0;
        });
    });
    q->wait();
}

void trainAndTestWithRegen(const int regen_steps = 0, const int retrain_steps = 0){
    
    cout << "Enter function with dim: " << ndims << std::endl;
    q = new queue(d_selector, e_handler);
    cout << "Hardware: " << q->get_device().get_info<info::device::name>() << std::endl;
    work_group_size = q->get_device().get_info<info::device::max_work_group_size>();
    cout << work_group_size << std::endl;
    cout << "reading files... " << std::endl;
	char *testFile = strdup(DATA_PATH "mnist_test.choir_dat");
	Data test = readData(testFile);
	fvec test_data = m2v(test.data);
	ivec test_labels = test.labels;

	char* trainFile = strdup(DATA_PATH "mnist_train.choir_dat");
	Data train = readData(trainFile);
    fvec train_data = m2v(train.data);
	ivec train_labels = train.labels;
    cout << "success!" << std::endl;
    
	nclasses = test.numClasses;
	nfeatures = test.numFeatures;

    cout << "allocate basis and class bufs" << std::endl;
    basisv = fvec(nfeatures*ndims, 0.0);
    basis_bufp = new fbuf(basisv);
    classesv = fvec(nclasses*ndims, 0.0);
    classes_bufp = new fbuf(classesv);

    cout << "generate basis: " << nfeatures << " " << ndims << std::endl;
    //generate basis
    tstart = TIME;

#if USE_ONEAPI_FUNCTIONS
    srand (time(NULL));
    int seed = rand();
    q->submit([&](auto &h){
		accessor acc(*basis_bufp, h, write_only);
		h.parallel_for(range(nfeatures*ndims), [=](auto index) {
			std::uint64_t offset = index.get_linear_id();
			oneapi::dpl::minstd_rand engine(seed, offset);
			oneapi::dpl::normal_distribution<float> distr;
			float res = distr(engine);
			acc[index] = res;
		});
	});
#else
    {
        host_accessor acc(*basis_bufp, write_only,no_init);
        static std::random_device dev;
        static std::mt19937 rng(dev());
        static std::normal_distribution<float> gen;
        for(int i = 0; i < nfeatures*ndims; i++) {
            acc[i] = gen(rng);
        }
    }
#endif
	q->wait();
    tend = TIME;
    cout << "generate basis time: " << elapsedTime(tstart, tend) << std::endl;

    int ndata_train = train_labels.size();
    fvec train_data_encoded(ndata_train * ndims, 0);
    fbuf train_data_buf(train_data);
    fbuf train_data_encoded_buf(train_data_encoded);
    int ndata_test = test_labels.size();
    fvec test_data_encoded(ndata_test * ndims, 0);
    fbuf test_data_buf(test_data);
    fbuf test_data_encoded_buf(test_data_encoded);
    tstart = TIME;
    encode(train_data_buf, train_data_encoded_buf, ndata_train);
    cout << "encode training data: " << elapsedTime(tstart, TIME) << std::endl;
    tstart = TIME;  
    encode(test_data_buf, test_data_encoded_buf, ndata_test);
    cout << "encode testing data: " << elapsedTime(tstart, TIME) << std::endl;

	int regen = regen_steps; // number of regen iterations
	int interim = retrain_steps; // number of iters between regens
	double percent_drop = 0.2; // how many dims to regen per regen iter
    
    //ibuf test_label_buf
    fvec testClasses(ndims * nclasses, 0.0);
    tstart = TIME;
    double fit_time = 0;
    double test_time = 0;
    double recalc_time = 0;
    test_times[0] = 0;
    test_times[1] = 0;
    FOR(i, regen){
        //cout << "regen #" << i << ", dims=" << ndims << "   ";
        int test_acc = 0, train_acc = 0;
        FOR(j, interim){
            test_acc = 0, train_acc = 0;
            auto t1 = TIME;

            train_acc = cpuFit(train_data_encoded_buf, train_labels, testClasses);
            fit_time += elapsedTime(t1, TIME);
            t1 = TIME;

            test_time += elapsedTime(t1, TIME);

            cout << (float) train_acc / (float) ndata_train << "\n";
            if (test_acc == ndata_test){
                i = regen;//super break
                break;
            }
        }
        auto t2 = TIME;
        cout << test_acc << "   " << train_acc << std::endl;
        //train data, test data, classes, basis
        int dim_loss = (float)ndims * (float)percent_drop;
        ivec dim_ranks = rankDims();
        dim_ranks.resize(dim_loss);
        updateClassesAndBasis(dim_ranks);
        encode(train_data_buf, train_data_encoded_buf, ndata_train);
        encode(test_data_buf, test_data_encoded_buf, ndata_test);
        recalc_time += elapsedTime(t2, TIME);
    }
    auto train_time = elapsedTime(tstart, TIME);
    cout << std::endl;
    cout << "training time: " << train_time << std::endl;
    cout << "     fit time: " << fit_time << std::endl;
    cout << "    test time: " << test_time << std::endl;
    cout << "  recalc time: " << recalc_time << std::endl;
    cout << std::endl;

    tstart = TIME;
    int accuracy = 0;
    testNN2(test_data_encoded_buf, test_labels, accuracy);
    accuracies.push_back((float)accuracy / (float)ndata_test);
    train_times.push_back((float)train_time);
    double inference_time = elapsedTime(tstart, TIME);
    cout << "Inference time: " << inference_time << std::endl;
    cout << "Accuracy: " << accuracy << " " << accuracies[accuracies.size()-1] << std::endl;
}
#endif

void trainAndTestOneShot(){
    
    cout << "Enter function with dim: " << ndims << std::endl;
    q = new queue(d_selector, e_handler);
    cout << "Hardware: " << q->get_device().get_info<info::device::name>() << std::endl;
    work_group_size = q->get_device().get_info<info::device::max_work_group_size>();

    cout << "reading files... " << std::endl;
	char *testFile = strdup(DATA_PATH "mnist_test.choir_dat");
	Data test = readData(testFile);
	fvec test_data = m2v(test.data);
	ivec test_labels = test.labels;

	char* trainFile = strdup(DATA_PATH "mnist_train.choir_dat");
	Data train = readData(trainFile);
    fvec train_data = m2v(train.data);
	ivec train_labels = train.labels;
    cout << "success!" << std::endl;
    
	nclasses = test.numClasses;
	nfeatures = test.numFeatures;

    cout << "classes, features: " << nclasses << " " << nfeatures << std::endl;

    cout << "allocate basis and class bufs" << std::endl;
    basisv = fvec(nfeatures*ndims, 0.0);
    basis_bufp = new fbuf(basisv);
    classesv = fvec(nclasses*ndims, 0.0);
    classes_bufp = new fbuf(classesv);

    cout << "generate basis: " << nfeatures << " " << ndims << std::endl;
    //generate basis
    tstart = TIME;
#if USE_ONEAPI_FUNCTIONS
    srand (time(NULL));
    int seed = rand();
    q->submit([&](auto &h){
		accessor acc(*basis_bufp, h, write_only);
		h.parallel_for(range(nfeatures*ndims), [=](auto index) {
			std::uint64_t offset = index.get_linear_id();
			oneapi::dpl::minstd_rand engine(seed, offset);
			oneapi::dpl::normal_distribution<float> distr;
			float res = distr(engine);
			acc[index] = res;
		});
	});
#else
    {
        host_accessor acc(*basis_bufp, write_only,no_init);
        static std::random_device dev;
        static std::mt19937 rng(dev());
        static std::normal_distribution<float> gen;
        for(int i = 0; i < nfeatures*ndims; i++) {
            acc[i] = gen(rng);
        }
    }
#endif
	q->wait();
    tend = TIME;
    cout << "generate basis time: " << elapsedTime(tstart, tend) << std::endl;


    int ndata_train = train_labels.size();
    fvec train_data_encoded(ndata_train * ndims, 0);
    fbuf train_data_buf(train_data);
    fbuf train_data_encoded_buf(train_data_encoded);
    int ndata_test = test_labels.size();
    fvec test_data_encoded(ndata_test * ndims, 0);
    fbuf test_data_buf(test_data);
    fbuf test_data_encoded_buf(test_data_encoded);
    tstart = TIME;
    encode(train_data_buf, train_data_encoded_buf, ndata_train);
    double encode_training_time = elapsedTime(tstart, TIME);
    cout << "encode training data: " << encode_training_time << std::endl;
    tstart = TIME;
    encode(test_data_buf, test_data_encoded_buf, ndata_test);
    cout << "encode testing data: " << elapsedTime(tstart, TIME) << std::endl;
    cout << train_data_encoded[0] << std::endl;
    cout << basisv[0] << std::endl;


    double fit_time = 0;
    double test_time = 0;
    int test_acc = 0, train_acc = 0, pre_train_acc = 0;

    tstart = TIME;
    fitOneShot(train_data_encoded_buf, train_labels, train_acc);
    fit_time += elapsedTime(tstart, TIME);
    cout << "training time: " << fit_time << std::endl;


    tstart = TIME;
    testNN2(test_data_encoded_buf, test_labels, test_acc);
    double inference_time = elapsedTime(tstart, TIME);
    cout << "testing time: " << inference_time << std::endl;
    cout << "correct guesses: " << test_acc << std::endl;
    cout << "testing acc: " << ((double) test_acc) / ((double)(ndata_test)) << std::endl;

    train_times.push_back((float)fit_time + (float)encode_training_time);
    accuracies.push_back((float)((double) test_acc) / ((double)(ndata_test)));
}

#if 0
void testInferenceBaseline(){
    
    cout << "Enter function with dim: " << ndims << std::endl;
    q = new queue(d_selector, e_handler);
    cout << "Hardware: " << q->get_device().get_info<info::device::name>() << std::endl;
    work_group_size = q->get_device().get_info<info::device::max_work_group_size>();

    cout << "reading files... " << std::endl;
	char *testFile = strdup(DATA_PATH "mnist_test.choir_dat");
	Data test = readData(testFile);
	fvec test_data = m2v(test.data);
	ivec test_labels = test.labels;

	char* trainFile = strdup(DATA_PATH "mnist_train.choir_dat");
	Data train = readData(trainFile);
    fvec train_data = m2v(train.data);
	ivec train_labels = train.labels;
    cout << "success!" << std::endl;
    
	nclasses = test.numClasses;
	nfeatures = test.numFeatures;

    cout << "classes, features: " << nclasses << " " << nfeatures << std::endl;

    cout << "allocate basis and class bufs" << std::endl;
    basisv = fvec(nfeatures*ndims, 0.0);
    basis_bufp = new fbuf(basisv);
    classesv = fvec(nclasses*ndims, 0.0);
    classes_bufp = new fbuf(classesv);

    cout << "generate basis: " << nfeatures << " " << ndims << std::endl;
    //generate basis
    tstart = TIME;
    srand (time(NULL));
    int seed = rand();
    q->submit([&](auto &h){
		accessor acc(*basis_bufp, h, write_only);
		h.parallel_for(range(nfeatures*ndims), [=](auto index) {
			std::uint64_t offset = index.get_linear_id();
			oneapi::dpl::minstd_rand engine(seed, offset);
			oneapi::dpl::normal_distribution<float> distr;
			float res = distr(engine);
			acc[index] = res;
		});
	});
	q->wait();
    tend = TIME;
    
    int ndata_train = train_labels.size();
    int ndata_test = test_labels.size();

    std::vector<double> times(10);
    FOR(time_i, times.size()){
        int batches = 500;
        int inputs = 1;
        std::vector<fvec> test_data;
        std::vector<fbuf> test_datab;
        ivec garbage_labels(inputs);
        FOR(i,batches) {
            test_data.push_back(fvec(inputs * ndims));
            test_datab.push_back(fbuf(test_data[i]));
        }
        tstart = TIME;
        int acc;
        FOR(i,batches){
            testNN2(test_datab[i], garbage_labels, acc);
        }
        double timed = elapsedTime(tstart, TIME);
        times[time_i] = timed;
        cout << "run #" << time_i+1 << "   time: " << timed << std::endl;
    }
    double average_time = 0;
    FORX(x,times) average_time += x/10.0;
    cout << "average: " << average_time << std::endl;
}
#endif
int main(int argc, char **argv){
	ndims = 2000;
	if(argc > 1) {
		ndims = atoi(argv[1]);
	}
    std::cout << "Starting inference only: " << ndims << std::endl;
    //trainAndTestWithRegen();
    trainAndTestOneShot();
    //testInferenceBaseline();
    return 0;
}