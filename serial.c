/*****************************************************************************/
// 
/*****************************************************************************/
// To compile serial_mmm:
// gcc -O1 serial_mmm.c -lrt -o serial_mmm
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define OPTIONS 3
#define IDENT 0

/* We want to test a wide range of matrix sizes, the sizes being
   used are defined below. */
int *matrix_sizes = {100, 500, 1000, 5000, 10000, 50000, 100000};

/* We also want to test a wide range of matrix densities, the
   densities being used are defined below. */
float *matrix_densities = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005,
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
  float density;
  data_t *values;
  int *columns;
  int *row_indices;
} csr_matrix, *csr_matrix_ptr;

/* Prototypes */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
csr_matrix_ptr new_csr_matrix(long int row_len);
int set__csr_matrix_row_length(csr_matrix_ptr m, long int row_len);
long int get_csr_matrix_row_length(csr_matrix_ptr m);
int init_csr_matrix(csr_matrix_ptr m, long int row_len, float matrix_density);
int f(matrix_ptr m, long int row_len);
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c);

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
  double time_stamp[OPTIONS][NUM_TESTS];
  double wakeup_answer;
  long int x, n, alloc_size;

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  printf("Dense MMM tests \n\n");

  wakeup_answer = wakeup_delay();

  printf("Doing MMM three different ways,\n");
  printf("for %d different matrix sizes from %d to %d\n",
                                                     NUM_TESTS, C, alloc_size);
  printf("This may take a while!\n\n");

  /* declare and initialize the matrix structure */
  matrix_ptr a0 = new_matrix(alloc_size);
  init_matrix(a0, alloc_size);
  matrix_ptr b0 = new_matrix(alloc_size);
  init_matrix(b0, alloc_size);
  matrix_ptr c0 = new_matrix(alloc_size);
  zero_matrix(c0, alloc_size);

  OPTION = 0;

  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_ijk(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
    set_matrix_row_length(a0, n);
    set_matrix_row_length(b0, n);
    set_matrix_row_length(c0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    mmm_kij(a0, b0, c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  if (OPTIONS > 2) {
    for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
      printf(" OPT %d, iter %ld, size %ld\n", OPTION, x, n);
      set_matrix_row_length(a0, n);
      set_matrix_row_length(b0, n);
      set_matrix_row_length(c0, n);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      mmm_jki(a0, b0, c0);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
  }

  printf("Done collecting measurements.\n\n");

  printf("row_len, ijk, kij, jki\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%ld, ", A*i*i + B*i + C);
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int) ((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
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
    

    return 1;
  }
  else return 0;
}

/* Zero COO matrix */
int zero_coo_matrix(coo_matrix_ptr m)
{
  if (row_len > 0) {
    m->len = row_len;
    m->nnz = 0;
    
    /* Remove non-zero elements */
    free(m->values);
    m->values = NULL;
    free(m->columns);
    m->columns = NULL;

    /* Clear row indices */
    for (int i = 0; i < row_len; i++) {
      m->row_indices[i] = IDENT;
    }

    return 1;
  }
  else return 0;
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
int init_csr_matrix(csr_matrix_ptr m, long int row_len)
{
  long int i;

  if (row_len > 0) {
    m->len = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      m->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

/* initialize csr matrix */
int zero_matrix(matrix_ptr m, long int row_len, )
{
  if (row_len > 0) {
    m->len = row_len;
    m->nnz = 0;
    
    /* Remove non-zero elements */
    free(m->values);
    m->values = NULL;
    free(m->columns);
    m->columns = NULL;

    /* Clear row indices */
    for (int i = 0; i < row_len; i++) {
      m->row_indices[i] = IDENT;
    }

    return 1;
  }
  else return 0;
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

/* csr mmm */
void mmm_csr(csr_matrix_ptr a, csr_matrix_ptr b, csr_matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_row_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for (i = 0; i < length; i++) {
    for (j = 0; j < length; j++) {
      sum = IDENT;
      for (k = 0; k < length; k++) {
        sum += a0[i*length+k] * b0[k*length+j];
      }
      c0[i*length+j] += sum;
    }
  }
}

/* mmm */
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_row_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (k = 0; k < length; k++) {
    for (i = 0; i < length; i++) {
      r = a0[i*length+k];
      for (j = 0; j < length; j++) {
        c0[i*length+j] += r*b0[k*length+j];
      }
    }
  }
}

/* mmm */
void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_row_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (j = 0; j < length; j++) {
    for (k = 0; k < length; k++) {
      r = b0[k*length+j];
      for (i = 0; i < length; i++) {
        c0[i*length+j] += a0[i*length+k]*r;
      }
    }
  }
}
