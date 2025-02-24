//
// Created by kazem on 2/23/25.
//

#ifndef IO_H
#define IO_H

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <string>
#include <sstream>


#define REALTYPE float

struct CSC {
 size_t m; // rows
 size_t n; // columns
 size_t nnz; // nonzeros
 int stype;
 bool is_pattern;
 bool pre_alloc; //if memory is allocated somewhere other than const.
 int *p; // Column pointer array
 int *i; // Row index array
 REALTYPE *x;
 CSC(size_t m, size_t n, size_t nnz, bool pre_alloc=false, int stype=0)
  : m(m), n(n), nnz(nnz), stype(stype), is_pattern(false), pre_alloc(pre_alloc) {
  p = new int[n+1]();
  i = new int[nnz]();
  x = new REALTYPE[nnz]();
 }
 };

 struct triplet{
  int row{}; int col{}; REALTYPE val{};
 };

 struct CSR{
  size_t m; // rows
  size_t n; // columns
  size_t nnz; // nonzeros
  int stype;
  bool is_pattern;
  bool pre_alloc; //if memory is allocated somewhere other than const.
  int *p; // Column pointer array
  int *i; // Row index array
  REALTYPE *x;
  CSR(size_t m, size_t n, size_t nnz, bool pre_alloc=false, int stype=0)
   : m(m), n(n), nnz(nnz), stype(stype), is_pattern(false), pre_alloc(pre_alloc) {
   p = new int[m+1]();
   i = new int[nnz]();
   x = new REALTYPE[nnz]();
  }
 };


//// NEW IO section. This will replace old read matrix gradually since this is more general.

  class missing_arg_error : public std::runtime_error
 {
 public:
  missing_arg_error (std::string arg, std::string msg="Argument missing")
   : std::runtime_error(msg)
   {
    arg_ = arg;
   }

  std::string arg() const { return arg_; }

 private:
  std::string arg_;
 };

 class open_file_error : public std::runtime_error
 {
 public:
  open_file_error (std::string filename, std::string msg="Failed to open file")
  : std::runtime_error(msg)
  {
   filename_ = filename;
  }

  std::string filename() const { return filename_; }

 private:
  std::string filename_;
 };

 class read_file_error : public open_file_error
  {
  public:
   read_file_error (std::string filename, std::string msg="Failed to read file")
   : open_file_error(filename, msg)
   {}
  };

 class write_file_error : public open_file_error
 {
 public:
  write_file_error (
    std::string filename,
    std::string msg="Failed to write to file"
  )
  : open_file_error(filename, msg)
  {}
 };

 class mtx_error : public std::runtime_error
 {
 public:
  mtx_error (std::string filename, std::string msg="Error loading matrix")
  : std::runtime_error(msg)
  {
   filename_ = filename;
  }

  std::string filename() const { return filename_; }

 private:
  std::string filename_;
 };

 class mtx_header_error : public mtx_error
 {
 public:
  mtx_header_error (
    std::string filename="Unknown",
    std::string msg="Invalid matrix header"
  )
  : mtx_error(filename, msg)
  {}
 };

 class mtx_format_error : public mtx_error
 {
 public:
  mtx_format_error
  (
    std::string expected_format,
    std::string got_format,
    std::string filename = "Unknown",
    std::string msg = "Matrix format mismatch"
  )
  : mtx_error(filename, msg)
  {
   expected_format_ = expected_format;
   got_format_ = got_format;
  }

  std::string expected_format() const { return expected_format_; }
  std::string got_format() const { return got_format_; }

 private:
  std::string expected_format_;
  std::string got_format_;
 };

 class mtx_arith_error : public mtx_error
 {
 public:
  mtx_arith_error
  (
    std::string expected_arith,
    std::string got_arith,
    std::string filename="Unknown",
    std::string msg="Matrix arithmetic mismatch"
  )
  : mtx_error(filename, msg)
  {
   expected_arith_ = expected_arith;
   got_arith_ = got_arith;
  }
  std::string expected_arith() const { return expected_arith_; }
  std::string got_arith() const { return got_arith_; }

 private:
  std::string expected_arith_;
  std::string got_arith_;
 };


#define print_precision  48
enum TYPE{
 REAL,INT,COMPLEX,PATTERN
};

enum SHAPE{// LOWER and UPPER both are symmetric matrices.
 LOWER,UPPER,GENERAL
};
enum FORMAT{
 COORDINATE,ARRAY
};

std::string type_str(int type) {
 switch (type) {
 case REAL:
  return "REAL";
 case INT:
  return "INT";
 case COMPLEX:
  return "COMPLEX";
 case PATTERN:
  return "PATTERN";
 default:
  return "UNKNOWN";
 }
}


std::string format_str(int fmt) {
 switch(fmt) {
 case COORDINATE:
  return "COORDINATE";
 case ARRAY:
  return "ARRAY";
 default:
  return "UNKNOWN";
 }
}

void trim(std::string &s) {
 s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
  return !std::isspace(ch);
 }));
 s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
  return !std::isspace(ch);
 }).base(), s.end());
}

void read_header(std::ifstream &inFile, int &n_row, int &n_col,
                 size_t &n_nnz, int &type, int &shape, int &mtx_format){
 std::string line,banner, mtx, crd, arith, sym;
 std::getline(inFile,line);
 trim(line);
 for (unsigned i=0; i<line.length(); line[i]=tolower(line[i]),i++);
 std::istringstream iss(line);
 if (!(iss >> banner >> mtx >> crd >> arith >> sym)){
  throw mtx_header_error("Unknown", "First line does not contain 5 tokens");
 }
 if(!(banner =="%%matrixmarket")) {
  throw mtx_header_error("Unknown", "first token is not \"%%%%MatrixMarket\"");
 }
 if(!(mtx =="matrix")) {
  throw mtx_header_error("Unknown", "Not a matrix, unable to handle");
 }
 if(crd == "coordinate") {
  mtx_format = COORDINATE;
 } else if(crd == "array") {
  mtx_format = ARRAY;
 } else{
  throw mtx_header_error("Unknown", "Unknown matrix format, unable to handle");
 }
 if(arith == "real")
  type = REAL;
 else if(arith == "integer")
  type = INT;
 else if (arith == "complex")
  type = COMPLEX;
 else if(arith == "pattern")
  type = PATTERN;
 else{
  throw mtx_header_error("Unknown",
                         "Unknown arithmetic, unable to handle");
 }
 if(sym == "symmetric")
  shape = LOWER;
 else if(sym == "general")
  shape = GENERAL;
 else{
  throw mtx_header_error("Unknown", "Unknown shape, unable to handle");
 }
 while (!line.compare(0,1,"%"))
 {
  std::getline(inFile, line);
  trim(line);
 }
 std::istringstream issDim(line);
 if(mtx_format != ARRAY){
  if (!(issDim >> n_row >> n_col >> n_nnz)){
   throw mtx_header_error("Unknown", "The matrix dimension is missing");
  }
 } else{
  if (!(issDim >> n_row >> n_col)){
   throw mtx_header_error("Unknown", "The matrix dimension is missing");
  }
  n_nnz = n_row*n_col;
 }
}



 void read_triplets_real(std::ifstream &inFile, int nnz,
                         std::vector<triplet>& triplet_vec,
                         bool read_val=true,
                         bool zero_indexing=false){
  for (int i = 0; i < nnz; ++i) {
   triplet tmp;
   inFile >> tmp.row;
   inFile >> tmp.col;
    if(read_val)
      inFile >> tmp.val;
    else
      tmp.val = 1.0;
   if(!zero_indexing){
    tmp.col--; tmp.row--;
   }
   triplet_vec.push_back(tmp);
  }
 }

int shape2int(int shape){
 int st = 0;
 switch (shape) {
 case LOWER:
  st=-1;
  break;
 case UPPER:
  st =1;
  break;
 case GENERAL:
  st = 0;
  break;
 default:
  st=0;
 }
 return st;
}

 void compress_triplets_to_csc(std::vector<triplet>& triplet_vec, CSC *A,
                               bool add_diags= true){
  assert(A->nnz == triplet_vec.size());
  std::sort(triplet_vec.begin(), triplet_vec.end(),
            [](const triplet& a, const triplet& b){return (a.col<b.col) || (a.col==b.col && a.row<b.row);});
  auto *count = new int[A->n]();
  for (auto i = 0; i < A->nnz; ++i) {
   count[triplet_vec[i].col]++;
  }
  A->p[0] = 0;
  for (auto j = 0; j < A->n; ++j) {
   if(count[j] == 0 && add_diags){ // insert zero diag for empty cols
    triplet tmp; tmp.col = tmp.row = j; tmp.val=0;
    triplet_vec.insert(triplet_vec.begin()+A->p[j], tmp);
    A->p[j+1] = A->p[j] + 1;
   }else{
    A->p[j+1] = A->p[j] + count[j];
   }
  }
  delete []count;
  for (auto k = 0; k < A->nnz; ++k) {
   A->i[k] = triplet_vec[k].row;
   A->x[k] = triplet_vec[k].val;
  }
 }


void read_mtx_csc_real(std::ifstream &in_file, CSC *&A, bool insert_diag){
 int n, m;
 int shape, arith, mtx_format;
 size_t nnz;
 std::vector<triplet> triplet_vec;

 read_header(in_file, m, n, nnz, arith, shape, mtx_format);
 if(arith != REAL && arith != INT && arith != PATTERN)
  throw mtx_arith_error("REAL", type_str(arith));
 if (mtx_format != COORDINATE)
  throw mtx_format_error("COORDINATE", format_str(mtx_format));
 bool read_val = true;
 if(arith == PATTERN)
  read_val = false;
 A = new CSC(m,n,nnz,false, shape2int(shape));
 read_triplets_real(in_file, nnz, triplet_vec, read_val);
 compress_triplets_to_csc(triplet_vec, A, insert_diag);
 A->nnz = A->p[n]; // if insert diag is true, it will be different.
 //print_csc(A->n, A->p, A->i, A->x);
}

void compress_triplets_to_csr(std::vector<triplet>& triplet_vec, CSR *A,
                               bool add_diags= true){
 assert(A->nnz == triplet_vec.size());
 std::sort(triplet_vec.begin(), triplet_vec.end(),
           [](const triplet& a, const triplet& b){return (a.row<b.row) || (a.row==b.row && a.col<b.col);});
 auto *count = new int[A->m]();
 for (auto i = 0; i < A->nnz; ++i) {
  count[triplet_vec[i].row]++;
 }
 A->p[0] = 0;
 for (auto j = 0; j < A->m; ++j) {
  if(count[j] == 0 && add_diags){ // insert zero diag for empty cols
   triplet tmp; tmp.col = tmp.row = j; tmp.val=0;
   triplet_vec.insert(triplet_vec.begin()+A->p[j], tmp);
   A->p[j+1] = A->p[j] + 1;
  }else{
   A->p[j+1] = A->p[j] + count[j];
  }
 }
 delete []count;
 for (auto k = 0; k < A->nnz; ++k) {
  A->i[k] = triplet_vec[k].col;
  A->x[k] = triplet_vec[k].val;
 }
}


void read_mtx_csr_real(std::ifstream &in_file, CSR *&A, bool insert_diag=true){
 int n, m;
 int shape, arith, mtx_format;
 size_t nnz;
 std::vector<triplet> triplet_vec;

 read_header(in_file, m, n, nnz, arith, shape, mtx_format);
 if(arith != REAL && arith != INT && arith != PATTERN)
  throw mtx_arith_error("REAL", type_str(arith));
 if (mtx_format != COORDINATE)
  throw mtx_format_error("COORDINATE", format_str(mtx_format));
 bool read_val = true;
 if(arith == PATTERN)
  read_val = false;
 A = new CSR(m,n,nnz,false, shape2int(shape));
 read_triplets_real(in_file, nnz, triplet_vec, read_val);
 compress_triplets_to_csr(triplet_vec, A, insert_diag);
 A->nnz = A->p[m]; // if insert diag is true, it will be different.
 //print_csc(A->n, A->p, A->i, A->x);
}

void spmv_csr(CSR &A, const REALTYPE *x, REALTYPE *y){
 for (int i = 0; i < A.m; ++i) {
  REALTYPE sum = 0;
  for (int j = A.p[i]; j < A.p[i+1]; ++j) {
   sum += A.x[j]*x[A.i[j]];
  }
  y[i] = sum;
 }
}

void get_mat_list(std::string file_path, std::string base_path, std::vector<std::string> &mat_list){
 std::ifstream in_file(file_path);
 if (!in_file.is_open()) {
  std::cout << "Error: Cannot open file " << file_path << std::endl;
  return;
 }
 std::string line;
 while (std::getline(in_file, line)) {
  mat_list.push_back(base_path + "/" + line);
 }
}

#endif //IO_H
