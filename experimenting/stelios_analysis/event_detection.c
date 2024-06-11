#include <stdio.h>
#include <stdlib.h>


#define SIZE 40 //equivalent to 2 windows of 20 samples each
#define HALF_SIZE ( SIZE / 2 )
#define ROLLED(IDX) ( (IDX < HALF_SIZE ? SIZE : 0) + IDX - HALF_SIZE )
#define CURR 1  // Index of current sum/average
#define PREV 0  // Index Previous sum/average
#define STEADY_SIZE (HALF_SIZE * 5)
#define MAX_REPS 1000
#define RANGE 1000
#define THRES 40.0  //define threshold in watts

#define MIN(A, B) ( A < B ? A : B )
#define ABS(X) ( X < 0 ? -X : X )



void clear_buff(float buf[])
    /* clear the buffer */
{
    for(int i=0; i<SIZE; i++) {
        buf[i] = 0.0;
    }
}
    

void eventoring(float raw, float buf[], float steady[])
/* Monitor events by checking rolling averages of streaming input*/
{
    static float sums[2] = { 0, 0 };
    float avgs[2] = { 0, 0 };
    float rolled;
    // if flag==0 event detection is ongoing. If flag==1 a 2nd buffer is filled with next 100 samples
    static unsigned int scount=0, flag=0, counter=0;   
    unsigned int index;
    

    index = counter++ % SIZE;

    if (!flag)    
    {
        /* gradually calculate averages*/
        
  
        rolled = buf[ROLLED(index)]; 
    
        /* add the new value, remove the replaced one */
        sums[CURR] += raw - rolled;
        
        if (counter <= SIZE){
            sums[PREV] += rolled - buf[index]; 
        }
        
        
        printf("%5u. buf[%u] was %.3f, becoming %.3f, rolled is buf[%u] = %.3f.\n", counter, index, buf[index], raw, ROLLED(index), rolled);
        
        buf[index] = raw;
        printf("%5u. Input is %.3f. Current sums are %.3f // %.3f. \n", counter, raw, sums[0], sums[1]);
        
        /*Check if buffer is full and modulo HALF_SIZE == 0*/
        if ((counter >= SIZE)&&(!(counter % HALF_SIZE)))
        {
            printf("checking averages...\n");
            for (int j=0; j<2; j++) {
                avgs[j] = sums[j] / HALF_SIZE;
            }
            
            if (ABS(avgs[0]-avgs[1]) > THRES) 
            {
                printf("Event detected. Averages differ by %.3f \n", avgs[0]-avgs[1]);
                flag=1;
            }
            
            /*every HALF_SIZE samples shift current sum to previous sum*/
            if (!(counter % HALF_SIZE)){
                sums[PREV] = sums[CURR];
            }
        }

        
    } else {
        // if steady buffer is not full, keep storing samples
        if (scount < STEADY_SIZE) 
        {
            //Store next 5 windows' samples
            steady[scount++] = raw;
        }
        else
        {
            // TODO: send data of both buffers to the cloud 
            flag=0; scount=0; counter=0; sums[0]=0; sums[1]=0;
            clear_buff(buf);
        }
    }

}




int main(int argc, char **argv)
{    
    unsigned int reps = MAX_REPS;
    float buf[SIZE]; // initialize buffer
    float steady[STEADY_SIZE]; //initialize a second buffer to store 5 windows worth of data
    
    clear_buff(buf);

    if (argc > 1) {
        reps = atoi(argv[1]);
    }

    int k;
    for(int j=0; j<reps; j++) {
//         eventoring(rand() % RANGE, buf, steady);
        eventoring(k, buf, steady);
        k++;
    }
    
    return 0;
}





