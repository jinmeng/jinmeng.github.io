---
layout: post
title: Extract all positives/negatives from a lmdb
---
本文介绍如何提取lmdb中所有的正样本或者负样本，并保存到另一个lmdb。

* content
{:toc}

## Basic steps:
*  Open source lmdb1 and target lmdb2 
*  Read a pair of key/value from lmdb1
*  If the key is a positive, then save this pair of key/value into lmdb2 and push the key into a vector  
*  Close both lmdbs
 
*  Save key vector into a file of list 
 
---

## code

```
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::pair;
using std::make_pair;
using std::string;
using std::ifstream;


//============================== MAIN function =================================================
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Read all positives from source lmdb, then write into target lmd and save all key strings into a file.\n"
        "Usage:\n"
        "    split_lmdb [FLAGS] str has_str SRC_DB TAR_DB TAR_DB_KEYLIST \n"
        "");
        
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    //FLAGS_log_dir="./log"; //jin: print both to console and log file NOW. set log_dir in split_lmdb.sh NOW 

    if (argc != 6) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/split_lmdb ");
        return 1;
    }

   
   const string str = argv[1], has_str = argv[2];
   const char*src_db = argv[3], *tar_db = argv[4];
   bool has = (has_str == "true");
   
   //jin: test
   //cout << "has = " << (has == true) << endl;
   //cout << "src_db = " << src_db << endl;
   //return 0;
   
//========================================================== 1.1 open source lmdb1
    MDB_env *mdb_env1; 
    MDB_dbi mdb_dbi1;  
    MDB_val mdb_key1, mdb_value1;  
    MDB_txn *mdb_txn1;  
    MDB_cursor *mdb_cursor1;  
   
    CHECK_EQ(mdb_env_create(&mdb_env1), MDB_SUCCESS) << "mdb_env_create failed";          
    CHECK_EQ(mdb_env_set_mapsize(mdb_env1, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";        
    CHECK_EQ(mdb_env_open(mdb_env1, src_db, MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";       
    CHECK_EQ(mdb_txn_begin(mdb_env1, NULL, MDB_RDONLY, &mdb_txn1), MDB_SUCCESS) << "mdb_txn_begin failed";      
    CHECK_EQ(mdb_open(mdb_txn1, NULL, 0, &mdb_dbi1), MDB_SUCCESS) << "mdb_open failed";        
    LOG(INFO) << "Opening source lmdb in " << src_db << endl;  
                  
	CHECK_EQ(mdb_cursor_open(mdb_txn1, mdb_dbi1, &mdb_cursor1), MDB_SUCCESS) << "mdb_cursor_open failed";
	CHECK_EQ(mdb_cursor_get(mdb_cursor1, &mdb_key1, &mdb_value1, MDB_FIRST), MDB_SUCCESS) << "mdb_cursor_get failed";

//========================================================== 1.2 open target lmdb2 

    MDB_env *mdb_env2;
    MDB_dbi mdb_dbi2;
    MDB_val mdb_key2;
    MDB_txn *mdb_txn2;       
    
    CHECK_EQ(mkdir(tar_db, 0744), 0) << "mkdir " << tar_db << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env2), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env2, 1099511627776), MDB_SUCCESS)  << "mdb_env_set_mapsize failed";// 1TB       
    CHECK_EQ(mdb_env_open(mdb_env2, tar_db, 0, 0664), MDB_SUCCESS)  << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env2, NULL, 0, &mdb_txn2), MDB_SUCCESS)   << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn2, NULL, 0, &mdb_dbi2), MDB_SUCCESS)  << "mdb_open failed. Does the lmdb already exist? ";
    LOG(INFO) << "Opening target lmdb " << tar_db;
    
//========================================================== 2.1. read each key strings from lmdb, 
    vector<string> tar_keystr_vec;
    
    int count = 0;     
    const int kMaxKeyLength = 256;
    
    LOG(INFO) << "Starting Iteration";
    do{       
        string keystr((char *)mdb_key1.mv_data, mdb_key1.mv_size);
           
//========================================================== 2.2 if the key is a positive, 
//then save this pair of key/value into lmdb2  and push the key into a vector

        //check if keystr is (or not) a substring of key of what we want; if so, remove its number id and push it back into keystr_vec
        if( has && (keystr.find(str) != string::npos) || !has && (keystr.find(str) == string::npos))  
        
        {                       
                
            unsigned tmp = keystr.find('_');
            string newkeystr = keystr.substr(tmp+1, keystr.size());
                                                                       				       
		    //convert keys in lmdb1 into keys in lmdb2 and put into lmdb2     
		    char key_cstr[kMaxKeyLength]; 
            snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count, newkeystr.c_str());          
                      
            string keystr2(key_cstr);
            tar_keystr_vec.push_back(keystr2);
            
            mdb_key2.mv_size = keystr2.size();
            mdb_key2.mv_data = reinterpret_cast<void*>(&keystr2[0]);
            CHECK_EQ(mdb_put(mdb_txn2, mdb_dbi2, &mdb_key2, &mdb_value1, 0), MDB_SUCCESS) << "mdb_put failed";
         
            
            if(++count % 10000 == 0)
            {
                CHECK_EQ(mdb_txn_commit(mdb_txn2), MDB_SUCCESS)  << "mdb_txn_commit failed";
                CHECK_EQ(mdb_txn_begin(mdb_env2, NULL, 0, &mdb_txn2), MDB_SUCCESS) << "mdb_txn_begin failed";
                
                LOG(ERROR) << "Processed " << count << " data.";
            }                 
        }
                 
    }while(mdb_cursor_get(mdb_cursor1, &mdb_key1, &mdb_value1, MDB_NEXT) == MDB_SUCCESS);

//========================================================== 3. write the last batch and close both lmdbs
    if (count % 1000 != 0) {                      
        CHECK_EQ(mdb_txn_commit(mdb_txn2), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(mdb_env2, mdb_dbi2);
        mdb_env_close(mdb_env2);
             
        mdb_close(mdb_env1, mdb_dbi1);
        mdb_env_close(mdb_env1);
        
        LOG(ERROR) << "Finally inserted " << count << " positives.";
    } 

//========================================================== 4. save key strings of lmdb2 into a file of list   
    std::ofstream outfile(argv[5]);
    if(!outfile)
    {
        cerr << "Failed to open file " << argv[5] << endl;
        return -1;
    }
    
    int keylist_vec_size = tar_keystr_vec.size();
    outfile << keylist_vec_size << endl;
    for(int i = 0; i < keylist_vec_size; ++i)
    {
        outfile << tar_keystr_vec[i] << endl;
    } 
                  
  return 0;

}
```


