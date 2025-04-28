/*****************************************************************************/
// 
/*****************************************************************************/
// To compile serial_mmm:
// gcc -O1 serial_sparse_mmm.c -lrt -o serial_sparse_mmm
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define NUM_SIZES 12
#define NUM_DENSITIES 7
#define IDENT 0
#define OPTIONS 2

/* We want to test a wide range of matrix sizes, the sizes being
   used are defined below. */
int matrix_sizes[NUM_SIZES] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000};

/* We also want to test a wide range of matrix densities, the
   densities being used are defined below. */
float matrix_densities[NUM_DENSITIES] = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005,
                           0.0001};

typedef float data_t;

/* It is not standard for sparse matrices to have standard 2D
   array data format, so we will test two different formats:
   coordinate list (COO) and compressed sparse row (CSR). */

/* COO format is defined as (row, column, value) tuples,
   ideally sorted first by row index, then by column index. */
typedef struct {
  int row;
  int column;
  data_t value;
} coo_element, *coo_element_ptr;

typedef struct {
  long int len;
  long int nnz; /* Number of non-zero elements. */
  float density;
  coo_element_ptr elements;
} coo_matrix, *coo_matrix_ptr;

/* CSR format is defined as three one dimensional arrays where
   each element has a value and a column, but the array
   representing rows indexes where in the other two arrays a
   given row starts. */
typedef struct {
  long int len;
  long int nnz; /* Number of non-zero elements. */
  long int nnz_allocated;
  float density;
  data_t *values;
  int *columns;
  int *row_indices;
} csr_matrix, *csr_matrix_ptr;

/* Prototypes */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
csr_matrix_ptr new_csr_matrix(long int row_len, float matrix_density);
int set_csr_matrix_row_length(csr_matrix_ptr m, long int row_len);
long int get_csr_matrix_row_length(csr_matrix_ptr m);
int init_csr_matrix(csr_matrix_ptr m);

int clock_gettime(clockid_t clk_id, struct timespec *tp);
coo_matrix_ptr new_coo_matrix(long int row_len, float density);
int set_coo_matrix_row_length(coo_matrix_ptr m, long int row_len);
long int get_coo_matrix_row_length(coo_matrix_ptr m);
int init_coo_matrix(coo_matrix_ptr m);

void mmm_csr(csr_matrix_ptr a, csr_matrix_ptr b, csr_matrix_ptr c);
void mmm_coo(coo_matrix_ptr a, coo_matrix_ptr b, coo_matrix_ptr c);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_SIZES][NUM_DENSITIES];
  double wakeup_answer;
  long int i, j, matrix_size;

  printf("Sparse MMM tests \n\n");

  wakeup_answer = wakeup_delay();

  printf("Starting CSR format MMM tests...\n");

  OPTION = 0;

  for (i=0; i<NUM_SIZES; i++) {
    matrix_size = matrix_sizes[i];
    for (j=0; j<NUM_DENSITIES; j++) {
      printf(" OPT %d, iter %ld, size %ld, density %f\n", OPTION, j + NUM_SIZES * i, matrix_size,
             matrix_densities[j]);
      if (matrix_size * matrix_size * matrix_densities[j] <= 2500000) {
	      csr_matrix_ptr a0 = new_csr_matrix(matrix_size, matrix_densities[j]);
	      csr_matrix_ptr b0 = new_csr_matrix(matrix_size, matrix_densities[j]);
	      csr_matrix_ptr c0 = new_csr_matrix(matrix_size, matrix_densities[j]);
	      init_csr_matrix(a0);
	      init_csr_matrix(b0);
	      init_csr_matrix(c0);
	      set_csr_matrix_row_length(a0, matrix_size);
	      set_csr_matrix_row_length(b0, matrix_size);
	      set_csr_matrix_row_length(c0, matrix_size);
	      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
              mmm_csr(a0, b0, c0);
              clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
              time_stamp[OPTION][i][j] = interval(time_start, time_stop);
      }
      else {
              time_stamp[OPTION][i][j] = 0;
      }
    }
  }
  
  OPTION++;
  for (i=0; i<NUM_SIZES; i++) {
    for (j=0; j<NUM_DENSITIES; j++) {
      matrix_size = matrix_sizes[i];
      printf(" OPT %d, iter %ld, size %ld, density %f\n", OPTION, j + NUM_SIZES * i, matrix_size,
             matrix_densities[j]);
      coo_matrix_ptr a0 = new_coo_matrix(matrix_size, matrix_densities[j]);
      coo_matrix_ptr b0 = new_coo_matrix(matrix_size, matrix_densities[j]);
      coo_matrix_ptr c0 = new_coo_matrix(matrix_size, matrix_densities[j]);
      init_coo_matrix(a0);
      init_coo_matrix(b0);
      init_coo_matrix(c0);
      set_coo_matrix_row_length(a0, matrix_size);
      set_coo_matrix_row_length(b0, matrix_size);
      set_coo_matrix_row_length(c0, matrix_size);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      mmm_coo(a0, b0, c0);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      time_stamp[OPTION][i][j] = interval(time_start, time_stop);
    }
  }
  
  printf("Done collecting measurements.\n\n");

  printf("Matrix density");
  for (int j = 0; j < NUM_SIZES; j++) {
    printf(",%d", matrix_sizes[j]); // %g for compact float printing
  }
  printf("\n");

  for (int i = 0; i < NUM_DENSITIES; i++) {
    printf("%f", matrix_densities[i]);
    for (int j = 0; j < NUM_SIZES; j++) {
        printf(",%ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[0][j][i])); // 0 for CSR
    }
    printf("\n");
  }
  printf("\n");

  printf("Wakeup delay computed: %g \n", wakeup_answer);
} /* End main */

/**********************************************/

/* Create COO matrix of specified length and density */
coo_matrix_ptr new_coo_matrix(long int row_len, float density)
{
  long int i;
  long int alloc;

  /* Allocate and declare header structure */
  coo_matrix_ptr result = (coo_matrix_ptr) malloc(sizeof(coo_matrix));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = row_len;
  result->density = density;
  result->nnz = (int) (row_len * row_len * density);

  /* Allocate and declare elements */
  if (result->nnz > 0) {
    alloc = result->nnz;
    coo_element_ptr data = (coo_element_ptr) calloc(alloc, sizeof(coo_element));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                                       alloc * sizeof(coo_element));
      return NULL;  /* Couldn't allocate storage */
    }
    result->elements = data;
  } else {
    result->elements = NULL;
  }

  return result;
}

/* Comparison function for qsort */
int compare_coo(const void *a, const void *b) {
  coo_element *ea = (coo_element *)a;
  coo_element *eb = (coo_element *)b;
  if (ea->row == eb->row) 
      return ea->column - eb->column;
  return ea->row - eb->row;
}

/* Initialize COO matrix */
int init_coo_matrix(coo_matrix_ptr m)
{
  long int i;

  if (m->nnz > 0) {
    for (i = 0; i < m->nnz; i++) {
      m->elements[i].value = (data_t)((data_t)rand() / (data_t)RAND_MAX);
      m->elements[i].row = (int) ((data_t)((data_t)rand() / (data_t)RAND_MAX) * m->len);
      m->elements[i].column = (int) ((data_t)((data_t)rand() / (data_t)RAND_MAX) * m->len);
    }
    

    /* Sort elements using qsort */
    qsort(m->elements, m->nnz, sizeof(coo_element), compare_coo);
    
    /* Remove duplicates */
    long int new_nnz = 0;
    for (long int i = 1; i < m->nnz; i++) {
        if (m->elements[i].row != m->elements[new_nnz].row ||
            m->elements[i].column != m->elements[new_nnz].column) {
            new_nnz++;
            m->elements[new_nnz] = m->elements[i];
        }
    }
    m->nnz = new_nnz + 1;
    
    return 1;
  }
  else return 0;
}

/* Convert sorted COO to CSR */
csr_matrix_ptr coo_to_csr(coo_matrix_ptr coo, csr_matrix_ptr csr) {
  if (coo->len != csr->len || coo->density != csr->density) {
    printf("ERROR: MATRIX DENSITIES OR LENGTHS DO NOT MATCH");
    return csr;
  }
  
  /* Build CSR structure */
  int current_row = -1;
  for (long int i = 0; i < coo->nnz; i++) {
      while (current_row < coo->elements[i].row) {
          current_row++;
          csr->row_indices[current_row] = i;
      }
      csr->columns[i] = coo->elements[i].column;
      csr->values[i] = coo->elements[i].value;
  }
  /* Finalize row pointers */
  csr->row_indices[coo->len] = coo->nnz;
  
  return csr;
}

/* For getting CSR format matrix elements */
coo_element_ptr get_coo_matrix_data_start(coo_matrix_ptr m)
{
  return m->elements;
}

/**********************************************/

/* Create csr matrix of specified length and density */
csr_matrix_ptr new_csr_matrix(long int row_len, float density)
{
  long int i;
  long int alloc;

  /* Allocate and declare header structure */
  csr_matrix_ptr result = (csr_matrix_ptr) malloc(sizeof(csr_matrix));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = row_len;
  result->density = density;
  result->nnz = (int) (row_len * row_len * density);
  result->nnz_allocated = (int)(row_len * row_len * density);

  /* Allocate and declare values */
  if (result->nnz > 0) {
    alloc = result->nnz;
    data_t *data = (data_t *) calloc(alloc, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                                       alloc * sizeof(data_t));
      return NULL;  /* Couldn't allocate storage */
    }
    result->values = data;
  } else {
    result->values = NULL;
  }

  /* Allocate and declare columns */
  if (result->nnz > 0) {
    alloc = result->nnz;
    int *data = (int *) calloc(alloc, sizeof(int));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                                       alloc * sizeof(int));
      return NULL;  /* Couldn't allocate storage */
    }
    result->columns = data;
  } else {
    result->columns = NULL;
  }

  /* Allocate and declare rows */
  if (result->len > 0) {
    alloc = result->len + 1;
    int *data = (int *) calloc(alloc, sizeof(int));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                                       alloc * sizeof(int));
      return NULL;  /* Couldn't allocate storage */
    }
    result->row_indices = data;
  } else {
    result->row_indices = NULL;
  }

  /* Allocate row indices with proper size */
    if (result->len > 0) {
        alloc = result->len + 1;  // Need len+1 row indices for CSR
        int *data = (int *) calloc(alloc, sizeof(int));
        if (!data) {
            free(result->values);
            free(result->columns);
            free(result);
            return NULL;
        }
        result->row_indices = data;
    }
    
    return result;
}

/* Set length of csr matrix */
int set_csr_matrix_row_length(csr_matrix_ptr m, long int row_len)
{
  m->len = row_len;
  return 1;
}

/* Return length of csr matrix */
long int get_csr_matrix_row_length(csr_matrix_ptr m)
{
  return m->len;
}

/* initialize csr matrix */
int init_csr_matrix(csr_matrix_ptr m) {
  coo_matrix_ptr coo = new_coo_matrix(m->len, m->density);
  if (!coo) return -1;
  
  if (!init_coo_matrix(coo)) {
      free(coo->elements);
      free(coo);
      return -1;
  }
  
  // Copy data from COO to CSR
  for (long int i = 0; i < coo->nnz; i++) {
      if (i < m->nnz) {  // Ensure we don't overflow
          m->values[i] = coo->elements[i].value;
          m->columns[i] = coo->elements[i].column;
      }
  }
  
  // Set row indices
  int current_row = -1;
  for (long int i = 0; i < coo->nnz; i++) {
      while (current_row < coo->elements[i].row) {
          current_row++;
          m->row_indices[current_row] = i;
      }
  }
  m->row_indices[m->len] = coo->nnz;
  
  free(coo->elements);
  free(coo);
  return 0;
}

/* For getting CSR format matrix element values */
data_t *get_csr_matrix_data_start(csr_matrix_ptr m)
{
  return m->values;
}

/* For getting CSR format matrix element columns */
int *get_csr_matrix_col_start(csr_matrix_ptr m)
{
  return m->columns;
}

/* For getting CSR format matrix row incdices */
int *get_csr_matrix_row_start(csr_matrix_ptr m)
{
  return m->row_indices;
}

/*************************************************/

int compare_int(const void *a, const void *b) {
  int int_a = *(const int *)a;
  int int_b = *(const int *)b;
  if (int_a < int_b) return -1;
  else if (int_a > int_b) return 1;
  else return 0;
}

/* csr mmm */
void mmm_csr(csr_matrix_ptr a, csr_matrix_ptr b, csr_matrix_ptr c)
{
  // Initialize dynamic storage for output matrix
    size_t capacity = 1024;  // Initial capacity
    c->values = realloc(c->values, capacity * sizeof(data_t));
    c->columns = realloc(c->columns, capacity * sizeof(int));
    c->nnz = 0;
    c->row_indices[0] = 0;

    data_t* temp = calloc(a->len, sizeof(data_t));
    int* temp_idx = malloc(a->len * sizeof(int));

    for (int i = 0; i < a->len; i++) {
        int nnz = 0;
        
        for (int j = a->row_indices[i]; j < a->row_indices[i+1]; j++) {
            int a_col = a->columns[j];
            data_t a_val = a->values[j];
            
            for (int k = b->row_indices[a_col]; k < b->row_indices[a_col+1]; k++) {
                int b_col = b->columns[k];
                if (temp[b_col] == 0) {
                    if (nnz >= a->len) {  // Check bounds
                        continue;
                    }
                    temp_idx[nnz++] = b_col;
                }
                temp[b_col] += a_val * b->values[k];
            }
        }

	// Resize if needed before adding new elements
        if (c->nnz + nnz > capacity) {
            while (c->nnz + nnz > capacity) 
                capacity *= 2;
            c->values = realloc(c->values, capacity * sizeof(data_t));
            c->columns = realloc(c->columns, capacity * sizeof(int));
        }
        
        // Add non-zero elements to output matrix
        qsort(temp_idx, nnz, sizeof(int), compare_int);
        for (int j = 0; j < nnz; j++) {
            int col = temp_idx[j];
            c->values[c->nnz] = temp[col];
            c->columns[c->nnz] = col;
            c->nnz++;
            temp[col] = 0;  // Reset for next row
        }
        c->row_indices[i+1] = c->nnz;
    }
    
    // Trim excess capacity
    c->values = realloc(c->values, c->nnz * sizeof(data_t));
    c->columns = realloc(c->columns, c->nnz * sizeof(int));

    free(temp);
    free(temp_idx);
}

/* mmm */
void mmm_coo(coo_matrix_ptr a, coo_matrix_ptr b, coo_matrix_ptr c)
{
  long int idx = 0;
  for (long int i = 0; i < a->nnz; i++) {
      int a_row = a->elements[i].row;
      int a_col = a->elements[i].column;
      data_t a_val = a->elements[i].value;
      
      for (long int j = 0; j < b->nnz; j++) {
          if (b->elements[j].row == a_col) {
              c->elements[idx].row = a_row;
              c->elements[idx].column = b->elements[j].column;
              c->elements[idx].value = a_val * b->elements[j].value;
              idx++;
          }
      }
  }
    
  c->nnz = idx;
  qsort(c->elements, c->nnz, sizeof(coo_element), compare_coo);
  
  /* Sum duplicates */
  long int new_nnz = 0;
  for (long int i = 1; i < c->nnz; i++) {
      if (c->elements[i].row == c->elements[new_nnz].row &&
          c->elements[i].column == c->elements[new_nnz].column) {
          c->elements[new_nnz].value += c->elements[i].value;
      } else {
          new_nnz++;
          c->elements[new_nnz] = c->elements[i];
      }
  }
  c->nnz = new_nnz + 1;
}

/* Set length of COO matrix */
int set_coo_matrix_row_length(coo_matrix_ptr m, long int row_len)
{
  m->len = row_len;
  return 1;
}

/* Return length of COO matrix */
long int get_coo_matrix_row_length(coo_matrix_ptr m)
{
  return m->len;
}