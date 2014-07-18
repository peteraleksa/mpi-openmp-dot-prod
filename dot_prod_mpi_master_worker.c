/* Parallel Dot Product in C w/ MPI using dynamic master-worker allocation
 * Peter Aleksa
 * peter.aleksa@gmail.com
 *
 */

 #include <math.h>                  // actually didn't use anything from here yet
 #include <mpi.h>                   // mpi library
 #include <stdio.h>                 // i/o
 #include <stdlib.h>                // for malloc

 #define DIETAG 1
 #define MASTER 0

 typedef int bool;
 #define true 1
 #define false 0

 int main(int argc, char* argv[]) {

 	MPI_Status status;						// MPI status variable
    double *x, *y;							// pointers to vectors to hold the vectors in the block
    double inner_prod; 						// inner prod variable
    double *local_results;					// array of local results
    double *results;					// array of global results
    int p; 									// the number of MPI processes
    int rank;								// the rank of the MPI process
   	int i;									// general counter variable
    int j;									// vector counter variable
    int block_size;						// size of each block of data sent and received
    int offset;							// data offset
    int results_offset;					// results offset
    int active;								// number of active workers
    int sender;								// sender process
    int vector_size;						// vector size
    int sample_size;						// data set size
    double time_elapsed;					// time elapsed variable
    int start;
    int end;
    
    char hostname[MPI_MAX_PROCESSOR_NAME];	// host process is running on
    int init_status;						// initialization error status flag
    bool initialized = false;				// mpi initialized flag
    int len;								// hostname length

    sample_size = 1000000000;               // this is the number of vectors
    vector_size = 15;                       // this is the size of each vector
    block_size = 10000;	                     // this is the block size of each chunk sent to workers

    /* initialize MPI */
    MPI_Initialized( &initialized );                    	 // set initialized flag
    if( !initialized )                                 		 // if MPI is not initialized
        init_status = MPI_Init( &argc, &argv );       		 // Initialize MPI
    else
        init_status = MPI_SUCCESS;   	              		      // otherwise set init_status to success
    if( init_status != MPI_SUCCESS ) {			     	         // if not successfully initialized
        printf ("Error starting MPI program. Terminating.\n");      // print error message
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, init_status);                     // abort
    }

    MPI_Get_processor_name( hostname, &len );                       // set hostname

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );                           // set process rank
    MPI_Comm_size( MPI_COMM_WORLD, &p );                            // set size of comm group

    //debug
    //printf("Process rank %d started on %s.\n", rank, hostname);     // print start message
    //fflush(stdout);

    /* Start Timer */
    MPI_Barrier( MPI_COMM_WORLD );                                  // synchronize
    time_elapsed = - MPI_Wtime();                                   // start time

    /* Master */
    if (rank == 0) {

    	active = 0;			// set active workers to 0
    	offset = 0;			// set data offset to 0
    	results_offset = 0;    // set results offset

    	local_results = (double *) malloc((block_size) * sizeof(double));
    	results = (double *) malloc((sample_size) * sizeof(double));

    	//printf("Master assigning blocks...\n");
    	//fflush(stdout);

    	/* assign first initial blocks of data to workers */
    	for (i = 1; i < p; i++) {
    		
            sender = i;
    		
            //printf("Master sending to process %i\n", sender);
    		//fflush(stdout);
    		
            // send a batch of data to the worker
    		MPI_Send(&offset, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
    		
            // increment active worker counter
    		active++;
    		
            // update the offset
    		offset += block_size;
    	
        }

    	/* receive results and assign more data if available */
    	do {
    		
    		// receive the local sum from any process
    		MPI_Recv(local_results, block_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    		
            // set sender variable to the process rank of the sender
    		sender = status.MPI_SOURCE;
    		
            //printf("Received results from process %i: %12.3f\n", sender, local_results[0]);
    		//fflush(stdout);
    		
            // add it to the global inner product vector
    		for (i = 0; i < block_size; i++) {
    			results[results_offset+i] = local_results[i];
    			//results_offset++;
    		}
    		
            // update the results offset
    		results_offset += block_size;
    		
            // decrement active worker counter
    		active--;

    		// if there is still data left to be processed
    		if (offset < sample_size) {
    			
                //printf("Master sending to process %i\n", sender);
    			//fflush(stdout);
    			
                // send the next batch of data to the worker
    			MPI_Send(&offset, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
    			
                // increment the active counter
    			active++;
    			
                // update the offset
    			offset += block_size;
    		
            }
    		
            // otherwise tell the worker to terminate
    		else {

    			//printf("Master sending terminate signal to process %i\n", sender);
    			//fflush(stdout);

    			// send terminate message to worker stored in sender
    			MPI_Send(&offset, 1, MPI_INT, sender, DIETAG, MPI_COMM_WORLD);
    		
            }

    	// repeat until all jobs have finished
    	} while(active > 0);

    	/* end timer and print message */
        time_elapsed += MPI_Wtime();
        
        //for (i = 0; i < sample_size; i++) {
        //	printf("%12.f\n",results[i]);
        //	fflush(stdout);
        //}
        //printf("The inner product is: %f\n", inner_prod);
        
        printf("Execution took %12.3f seconds\n", time_elapsed);
        fflush(stdout);

    }

    /* Workers */
    else {

    	x = (double *) malloc(block_size * vector_size * sizeof(double));
    	y = (double *) malloc(block_size * vector_size * sizeof(double));
    	local_results = (double *) malloc(block_size * sizeof(double));

    	// start work loop
    	while(true) {

    		// receive offset from master
    		MPI_Recv(&offset, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    		//printf("Process %i received data chunk\n", rank);
    		//fflush(stdout);

    		// if there is a terminate message then break
    		if (status.MPI_TAG == DIETAG)
    			break;

    		// read current block of the vectors starting from offset
    		for (i = 0; i < block_size * vector_size; i++) {
    			x[i] = 1.0;		// dummy data
    			y[i] = 1.0;		// dummy data
    		};

    		// for each vector in the block
    		for (i = 0; i < block_size; i++) {
    			// zero out inner product variable
    			inner_prod = 0.00;
    			// set start position
    			start = i * vector_size;
    			// set end position
    			end = start + vector_size;
    			// calculate the dot product
    			for (j = start; j < end; j++) {
    				inner_prod += x[j] * y[j];
    			}
    			// save the result to the array of local results
    			local_results[i] = inner_prod;
    		}

    		//printf("Process %i sent data to master\n", rank);
    		//fflush(stdout);
    		
            // send the results to the master
    		MPI_Send(local_results, block_size, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);

    	}

    }

    //printf("Finalizing...\n");
    //fflush(stdout);
    
    /* finalize MPI */
    MPI_Finalize();

 	exit(0);
 }