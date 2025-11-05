// Exemplo de ordenacao de inteiros com thrust
//   com pequenas modificacoes 
//   de w.zola para medir tempo com o pacote chrono.h

//
//   OBS: fazendo para 32Milhoes de elementos
//        rodando o kernel 30 vezes e tirando a media

 #include "chrono.c"
 #define N_REPETICOES 1 //30

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>

#include <chrono>    // usaremos o chrono de g++ para comparação também
                     //  (apenas para teste)

int main(void)
{

        printf("\n----- exemplo com thrust::sort para inteiros\n"
               "          modificado com medicoes de tempo\n"
               "          usando pacote chrono.c e calculando \n"
               "          vazão de numeros inteiros de 32Bits ordenados por segundo\n"
               "------------------------------------------\n\n\n" );
     
     int nElements = 32 * 1000 * 1000; // 32Million elements
               
     // generate nElements random numbers on the host
     thrust::host_vector<int> h_vec( nElements );
     thrust::generate( h_vec.begin(), h_vec.end(), rand );

     // transfer data to the device
     thrust::device_vector<int> d_vec = h_vec;

    printf( "Will sort %d integers\n", nElements );
    
    cudaDeviceSynchronize();
    chronometer_t chrono_para_Sort;  // cria um chonometro para medir thrust::sort
    chrono_reset( &chrono_para_Sort );
    
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    chrono_start( &chrono_para_Sort );
       for (int i = 0; i < N_REPETICOES; ++i) {
            // sort data on the device
            thrust::sort(d_vec.begin(), d_vec.end());
       }
    cudaDeviceSynchronize();
    chrono_stop( &chrono_para_Sort );
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    
    printf( "\n----- reportando o tempo total para\n"
            "as %d ativações do kernel Sort do thrust -------",
                   N_REPETICOES );
    chrono_reportTime( &chrono_para_Sort, (char *)"thrust::sort kernel" );
    
    
    printf("\n\n" );    

    std::cout << "Tempo (medido na stdlib) = " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
              << "ms" << std::endl;

    // calcular e imprimir a VAZAO (numero de INT/s)
    double total_time_in_seconds = (double) chrono_gettotal( &chrono_para_Sort ) /
                                      ((double)1000*1000*1000);
    printf( "total_time_in_seconds: %lf s, for %d repetitions\n", total_time_in_seconds, N_REPETICOES );
    printf( "time_in_seconds: %lf s, for EACH activation of sort\n", total_time_in_seconds/N_REPETICOES );
    
                                  
    double OPS = ((double)nElements*N_REPETICOES)/total_time_in_seconds;
    
    
    
    printf( "Throughput: %lf INT/s\n", OPS );
    
   

     // transfer data back to host
     thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
     
     printf("\n----- OBS:\n"
               "     essa experiencia pode nao estar reportando\n"
               "     corretamente a vazão pois na segunda ordenação\n"
               "     até a última os dados já estarão ordenados\n"
               "     podendo influenciar o tempo gasto\n"
               "------------------------------------------\n\n\n" );

     return 0;
}
