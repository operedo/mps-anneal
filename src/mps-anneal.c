/**
 * \file	mps-anneal.c 
 * \brief 	Programa que realiza el Recocido Simulado (Simulated Annealing) para la estimacion
 * 		de estadisticas de multiples puntos (MPS) utilizando Computacion Paralela, bajo el formato del estandar MPI. 
 * \author	Oscar Francisco Peredo Andrade operedo@gmail.com
 * \date	30 de Mayo del 2008
 * \version	1.0
 * 
 * El desarrollo de esta implementacion se enmarca en el proyecto FONDECYT-REGULAR 1061260 titulado 
 * "Evaluacion de yacimientos mediante simulacion esticastica integrando estadisticas de multiples puntos". La implementacion se probo en
 * el Laboratorio de Planificacion Minera, el cual cuenta con un cluster Rocks, que permite ingresar trabajos a un cola para ser procesados
 * en algun momento por las maquinas. 
 *
 * Para mayores detalles contactar a Oscar Peredo (operedo@gmail.com) o Julian Ortiz (jortiz@ing.uchile.cl).
 *
 * Para compilar este archivo se debe ejecutar, desde el directorio mps-anneal:
 * \code
 * $ mpic++ -g -o bin/mps-anneal src/mps-anneal.c 
 * \endcode
 * Se incluye un Makefile con el cual se puede setear el directorio de salida (bin) y entrada (src), ademas de otras librerias que se deseen utilizar:
 * \code
 * $ make
 * \endcode
 * Para ejecutar, se utiliza el siguiente script, llamado execute_mps-anneal.sh:
 * \code
 * #!/bin/bash
 * #$ -cwd
 * #$ -j y
 * #$ -m abes 
 * #$ -N mps-anneal
 * #$ -notify
 * #$ -pe mpi 7
 * #$ -S /bin/bash
 * #$ -now no
 * set -- $(date "+%Y %m %d %H %M %S %Z")
 * year=$1
 * month=$2
 * day=$3
 * hour=$4
 * min=$5
 * sec=$6
 * tz=$7
 * timestamp="${month}${day}${hour}${min}"
 * nx=100
 * ny=100
 * nz=1
 * tnx=2
 * tny=2
 * tnz=1
 * scenario="B"
 * resources="/home/operedo/mps-anneal/resources"
 * dataTI="${resources}/test${nx}x${ny}x${nz}.dat"
 * dataRE="${resources}/randomimage${nx}x${ny}x${nz}.dat"
 * logTI="${resources}/logTI_${nx}x${ny}x${nz}_${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}.txt"
 * logRE="${resources}/logRE_${nx}x${ny}x${nz}_${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}.txt"
 * topo="${resources}/topology${NSLOTS}.dat"
 * out="${resources}/${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}.dat"
 * sched="${resources}/schedule${nx}x${ny}x${nz}_${tnx}x${tny}x${tnz}_${NSLOTS}_${scenario}_${timestamp}.dat"
 * rand="${resources}/randomfile${nx}x${ny}x${nz}_${scenario}.dat"
 * rand2="${resources}/randomfile${nx}x${ny}x${nz}_${scenario}_2.dat"
 * hashsize=400000
 * t0=5000
 * lambda=0.1
 * npert=$[1000 * $nx * $ny *$nz]
 * ntemp=10
 * maxatt=5
 * irepo=$[1*$nx*$ny*$nz]
 * advance="salida${tnx}x${tny}x${tnz}_${NSLOTS}_${timestamp}_${scenario}_${t0}.dat"
 * mpirun -np $NSLOTS bin/mps-anneal 	$dataTI $dataRE $logTI $logRE $topo $out $sched $rand $rand2 $nx $ny $nz \,
 * 					$tnx $tny $tnz $hashsize $t0 $lambda $npert $ntemp $maxatt $irepo \,
 *					> salidas/$advance
 * exit 0
 * \endcode
 * Este script se ejecuta de la siguiente manera:
 * \code
 * $ qsub execute_mps-anneal.sh
 * Your job 227 ("mps-anneal") has been submitted
 * \endcode
 * Para revisar la cola de trabajos ingresados al cluster:
 * \code
 * $ qstat
 * job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID 
 * -----------------------------------------------------------------------------------------------------------------
 *    227  0.00000 mps-anneal operedo      qw    06/03/2008 06:02:16                                    7        
 * \endcode
 * Para eliminar un trabajo de la cola:
 * \code
 * $ qdel 227
 * \endcode
 * La linea que indica cuantos procesos se utilizaran es la siguiente: 
 * \code  
 * #$ -pe mpi 7 
 * \endcode
 * El resto de las lineas describen el valor y la ubicacion de los parametros que se entregan al programa.
 */

#include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <ctime>
#include <map>
#include <string>


#define WHITE '/'
#define BLACK 'P'
#define GRAY '\000'
#define RELEVANT 'b'
#define NOT_RELEVANT 'N'

#define DEBUG 0
#define DEBUG_AC 0
#define DEBUG_RE 0
#define DEBUG_OUT 0
#define DEBUG_PRINT 0
#define DEBUG_END 1

#define INVERSEWEIGHT 0

using namespace std;


/**
 * \struct block
 * \brief Estructura que almacena la informacion de un punto de la grilla. Cuando se hable de bloque o punto, se entendera por un elemento
 * asociado a esta estructura.
 */
struct block{
	int x;
	int y;
	int z;
	char data;
};

/**
 * \struct geometry
 * \brief Estructura que almacena la informacion de la grilla de puntos. Tambien se utiliza para almacenar el template que generara los patrones
 * en la simulacion.
 */
struct geometry{
	int lengthx;
	int lengthy;
	int lengthz;
	block ***node;
};

/*funciones asociadas a utilidades*/
long filesize(char *);
int level(int);

/*funciones asociadas a datos*/
void load_template(geometry *);
void free_template(geometry *);
void reset_template(geometry *);
void load_associatedPatterns(geometry *, geometry *,unsigned int,map<string,int>&, ofstream&);
void evaluate_template(int, int, int, block ,int, geometry *, geometry *,unsigned int,map<string,int>&, ofstream&);
string evaluate_template_ret(int, int, int, int *, int,geometry*, geometry*,map<string,int>&);
void load_reservoir(geometry *,geometry *,char *,double);
void copy_reservoir(geometry *, geometry *);
void print_reservoir(geometry *,char *);
void generate_reservoir(geometry *,double);
void free_reservoir(geometry *);
void free_reservoir_nodes(geometry *);
void open_logfile(ofstream&  ,char *);
void print_to_log(ofstream& ,char *);
void close_logfile(ofstream&);
int *load_topology_dims(char *);
void load_topology(char *, int *, int *);
void load_random(char *, float *);
void load_random_3cols(char *, int *, int*, int*);
void print_histogram(ofstream&,map<string,int>&);

/*funciones asociadas al recocido simulado*/
void schedule_update(int, int , double , double* , int* , int , double* , double ,int*,int,int,int* , map<string,int>& ,map<string,int>& , geometry *, char *,ofstream &, char *,float);
void print_schedule(double , int , double , int , char * );
void load_schedule(double *, int *, double *, int *, char *);
float total_weight(map<string,int>&,float);
int* random_block(geometry*,int,int,int*,int*,int*);
double evaluate_realization(map<string,int>&, map<string,int>&,float);
void perturb_realization(geometry*, geometry*, int *, map<string,int>&, int, int);
int decide_perturbation(double,double,double,int,int,int,float*);


/**
 * \brief Funcion principal. Realiza la lectura de los argumentos y la iteraci\'on principal del Recocido Simulado en Paralelo.
 * \param argc	Numero de argumentos. Deben ser exactamente 23.
 * \param argv	Los argumentos son los siguientes:
 *		- Archivo con grilla asociada a imagen de entrenamiento.
 *		- Archivo con grilla asociada a una realizacion (inicial).
 * 		- Archivo donde se guardara el histograma de frecuencias asociadas a un template, encontradas 
 *		en la imagen de entrenamiento, en formato "columna-1,columna-2", con columna-1 el id del patron y columna-2 la frecuencia (entero>=0).
 * 		- Archivo donde se guardara el histograma de frecuencias asociadas a un template, encontradas 
 *		en la realizacion, en formato "columna-1,columna-2", con columna-1 el id del patron y columna-2 la frecuencia (entero>=0).
 * 		- Archivo donde se especifica la topologia del arbol binario que se utilizara para la paralelizacion.
 * 		- Archivo donde se guardara la grilla asociada a la realizacion final.
 * 		- Archivo donde se guardara el schedule utilizado en cada iteracion. solo se utiliza cuando hay varios procesos corriendo. 
 * 		- Archivo con puntos aleatorios de la grilla. Se utiliza para tener un camino predeterminado de perturbaciones.
 * 		- Archivo con valores aleatorios entre 0 y 1. Se utiliza para tener un camino predeterminado de numeros aleatorios comun a todos 
 *		los procesos.
 * 		- Dimension x de la grilla.
 * 		- Dimension y de la grilla.
 * 		- Dimension z de la grilla.
 * 		- Dimension x del template.
 * 		- Dimension y del template.
 * 		- Dimension z del template.
 * 		- Tamano maximo que puede alcanzar el map donde se almacenan las frecuencias. No se utiliza, pero se implemento por posibles usos futuros.
 * 		- Tamano maximo del buffer que almacenara los id's de los patrones asociados a un template.
 * 		- Temperatura inicial.
 * 		- Factor de reduccion lambda.
 * 		- Numero maximo de perturbaciones.
 * 		- Numero maximo de repeticiones.
 * 		- Numero maximo de intentos, sin exito, para reducir la funcion objetivo. Cuando se alcanza el maximo, se reduce la temperatura.
 * 		- Numero de iteraciones que deben ocurrir para realizar un reporte del estado.
 *
 *
 * En esta funcion se implementa el Recocido Simulado a modo de prueba, para chequear la efectividad del speedup teorico \f$\log_2(P+1)\f$. Puede ser
 * ejecutado con \f$P\f$ procesos, donde \f$P\f$ debe satisfacer que \f$\log_2(P+1)\f$ es un numero natural. Esta condicion se debe a que solo se 
 * consideraron arboles binarios para la Computacion Especulativa.
 */
int main(int argc, char **argv) {
	long begin=time(NULL);
	cout << "BEGIN: " <<  begin<< endl;
	
	int my_id;
	int master=0;
	int num_procs;
	MPI_Comm comm;
	MPI_Request req;
	
	int *index=NULL;
	int *edges=NULL;	
	int *dims;

	int *rand1;
	int *rand2;
	int *rand3;
	float *randarray2;

	geometry* tem=new geometry;
	geometry* trainingImage=new geometry;
	geometry* realization=new geometry;

	map<string,int> hash;
	map<string,int> hashrealization;
	map<string,int> hashaux;

	ofstream mylog,mylog2;
	unsigned int HASH_SIZE;
	int BUFFER_SIZE;
	if(argc!=23){
		printf("Use:\n./mps-anneal datafile datafile2 logfile logfile2 topofile outfile schedfile nx ny nz tnx tny tnz hashsize t0 lambda npert ntemp maxatt irepo\n");
		exit(1);
	}
	char * filename=argv[1];
	char * filename2=argv[2];
	char * logfilename=argv[3];
	char * logfilename2=argv[4];
	char * topofile=argv[5];
	char * outfile=argv[6];
	char * schedulefile=argv[7];
	char * randomfile=argv[8];
	char * randomfile2=argv[9];
	trainingImage->lengthx=atoi(argv[10]);
	trainingImage->lengthy=atoi(argv[11]);
	trainingImage->lengthz=atoi(argv[12]);
	tem->lengthx=atoi(argv[13]);
	tem->lengthy=atoi(argv[14]);
	tem->lengthz=atoi(argv[15]);

	HASH_SIZE=atoi(argv[16]);
	BUFFER_SIZE=tem->lengthx*tem->lengthy*tem->lengthz;

	realization->lengthx=atoi(argv[10]);
	realization->lengthy=atoi(argv[11]);
	realization->lengthz=atoi(argv[12]);

	double T=atof(argv[17]);
	double T0=atof(argv[17]);
	double lambda=atof(argv[18]);
	int npert=atoi(argv[19]);
	int ntemp=atoi(argv[20]);
	int maxatt=atoi(argv[21]);
	int irepo=atoi(argv[22]);

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_id);

	dims=load_topology_dims(topofile);
	if(dims[0]>0 && dims[1] >0){
		index=new int[dims[0]];
		edges=new int[dims[1]];
		cout << "dim_index=" << dims[0] << " dim_edges=" << dims[1] << endl;
		load_topology(topofile,index,edges);
	}
	load_template(tem);

	/*load training image*/
	load_reservoir(trainingImage,tem,filename,0.5);
	open_logfile(mylog,logfilename);
	load_associatedPatterns(tem,trainingImage,HASH_SIZE,hash,mylog);

	cout << "TI: " << filename << endl;

	close_logfile(mylog);
	free_reservoir(trainingImage);
	
	
	/*load random image*/
	load_reservoir(realization,tem,filename2,0.5);
	open_logfile(mylog2,logfilename2);
	load_associatedPatterns(tem,realization,HASH_SIZE,hashrealization,mylog2);
	cout << "RE: " << filename2 << endl;
	
	close_logfile(mylog2);	

	/*balanced tree*/
	if(num_procs>1){
		//cout << "index=" << *index << ", edges=" << *edges << endl;
		MPI_Graph_create(MPI_COMM_WORLD,num_procs,index,edges,0,&comm);
		cout << "paso graph_create" << endl;
		int topo_type;
		MPI_Topo_test( comm, &topo_type );
		cout << "paso topo_test" << endl;
		if (topo_type != MPI_GRAPH) {
			printf( "Topo type of comm was not graph\n" );
			fflush(stdout);
		}

	}
	int num_levels=(int)(log(num_procs)/log(2))+1;
	cout << "num_levels=" << num_levels << endl;

	// least common multiple
	//int irepofix=((int)floor(((double)irepo)/((double)12)))*12; /*12 es divisible por 2,3 y 4, profundidades asociadas a 3,7 y 15 procesadores*/
	int irepofix=((int)floor(((double)irepo)/((double)840)))*840; /*840 es divisible por 2,3,4,5,6,7,8, profundidades asociadas a 3,7,15,31,63,127,255 procesadores*/
	//int irepofix=((int)floor(((double)irepo)/((double)2520)))*2520; /*2520 es divisible por 2,3,4,5,6,7,8,9,10 profundidades asociadas a 3,7,15,31,63,127,255,511,1023 procesadores*/


	rand1=new int[npert];
	rand2=new int[npert];
	rand3=new int[npert];
	load_random_3cols(randomfile, rand1, rand2, rand3);
	randarray2=new float[npert];
	load_random(randomfile2,randarray2);

	int lastupdater=0;
	int tag=0;
	int iter=0;
	int attempt=0;
	int tred=0;
	float prom=total_weight(hash,1.0);
	double obj=evaluate_realization(hash,hashrealization,prom);
	double startobj=obj;
	double latestobj=obj;
	double objprevt=1.0e32;

	int maxrepetitions=maxatt*2;
	int repcounter=0;
	
	while(iter<npert && tred<ntemp){
		tag=iter;
		if(my_id==master){
			schedule_update(iter,irepofix,startobj,&latestobj,&attempt,maxatt,&T,lambda,&tred,ntemp, maxrepetitions, &repcounter,hash,hashrealization, realization,outfile,mylog2, logfilename2,prom);
			print_schedule(latestobj,attempt,T,tred,schedulefile);
		}	

		if(num_procs>1){
			MPI_Barrier(comm);
			load_schedule(&latestobj, &attempt, &T, &tred, schedulefile);
		}

			int *perturbations=new int[4*num_levels];	
			if(my_id==master){/*MASTER*/
				int *rejector,*accepter;
				rejector=new int;
				accepter=new int;
				int *localarray=random_block(realization,my_id,iter,rand1,rand2,rand3);
				int *nomove=new int[3];
				nomove[0]=-1;
				nomove[1]=-1;
				nomove[2]=-1;
				if(DEBUG)cout << "(" << iter << ")MASTER perturbing data: " << localarray[0] << "," << localarray[1] << "," << localarray[2] << endl;
				if(num_procs>1){
					int *numneigh=new int;
					MPI_Graph_neighbors_count(comm,my_id,numneigh);
					int *neigh=new int[*numneigh];
					MPI_Graph_neighbors(comm,my_id,*numneigh,neigh);
					MPI_Send(localarray,3,MPI_INT,neigh[0],tag,comm);
					MPI_Send(nomove,3,MPI_INT,neigh[1],tag,comm);
					delete [] neigh;
					delete numneigh;
				}
				double *old_value=new double;
				*old_value=evaluate_realization(hash,hashrealization,prom);
				if(DEBUG)cout <<"("<<iter<<")"<< "MASTER calculate old value " << *old_value << endl;
				perturb_realization(tem,realization,localarray,hashrealization,my_id,iter);
				double *new_value=new double;
				*new_value=evaluate_realization(hash,hashrealization,prom);
				if(DEBUG)cout <<"("<<iter<<")"<< "MASTER calculate new value " << *new_value << endl;
				
				int *decision=new int;
				*decision=decide_perturbation(*old_value, *new_value,T,my_id, iter,num_procs,randarray2 );
				if(*decision){
					if(DEBUG)cout <<"("<<iter<<")"<< "MASTER accept perturbation" << endl;
					*accepter=1;
					*rejector=2;
				}
				else{
					if(DEBUG)cout <<"("<<iter<<")"<< "MASTER reject perturbation" << endl;
					perturb_realization(tem,realization,localarray,hashrealization,my_id,iter);
					*accepter=2;
					*rejector=1;
				}
				
				perturbations[0]=localarray[0];
				perturbations[1]=localarray[1];
				perturbations[2]=localarray[2];
				perturbations[3]=*decision;
				
				if(num_procs>1){
					if(DEBUG)cout <<"("<<iter<<")"<< "MASTER sending signal GO to processor "<<*accepter<<" and signal STOP to processor "<<*rejector << endl;
					int *go=new int;
					int *stop=new int;
					*go=1;
					*stop=0;
					MPI_Send(go,1,MPI_INT,*accepter,tag,comm);
					MPI_Send(stop,1,MPI_INT,*rejector,tag,comm);
					delete go;
					delete stop;


					int *modif=new int[4*(num_levels-1)];
					int *coord1=new int[3];
					if(DEBUG)cout <<"("<<iter<<")"<<"MASTER waiting for receive info from processor " << *accepter << endl;
					MPI_Status *status0=new MPI_Status;
					MPI_Recv(modif,4*(num_levels-1),MPI_INT,*accepter,tag,comm,status0);
					delete status0;
					int *i,*t;
					i=new int;
					t=new int;
					if(DEBUG){
						cout <<"("<<iter<<")"<<"MASTER received info: ";
						for(*i=0;*i<4*(num_levels-1);(*i)++){
							if(*i==4*(num_levels-1)-1)
								cout << modif[*i] << " from processor "<< *accepter << endl;
							else
								cout << modif[*i] << ",";
						}
					}
					for(*i=0;*i<4*(num_levels-1);(*i)++){
						perturbations[4+*i]=modif[*i];
					}

					if(*decision){
                                                perturb_realization(tem,realization,localarray,hashrealization,my_id,iter);
                                        }

					delete [] modif;
					delete [] coord1;
					delete i;
					delete t;
				}
				delete [] localarray;
				delete [] nomove;
				delete decision;
				delete accepter;
				delete rejector;
				delete old_value;
				delete new_value;
			}
			else{/*SLAVE*/
				/*update changes*/

				int actuallevel=level(my_id);
				int *signal=new int;	
				int *rejector,*accepter;
				rejector=new int;
				accepter=new int;
				int *localarrayrecv=new int[3*(actuallevel-1)];
				int *localarraypert=new int[3];
				int *localarraysend;
				int *nomove=new int[3];
				nomove[0]=-1;
				nomove[1]=-1;
				nomove[2]=-1;
				int *numneigh=new int;
				MPI_Graph_neighbors_count(comm,my_id,numneigh);
				int *neigh=new int[*numneigh];
				MPI_Graph_neighbors(comm,my_id,*numneigh,neigh);
				if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" waiting for receive from processor " << neigh[0] << endl;
				

					MPI_Status *status1=new MPI_Status;
					MPI_Recv(localarrayrecv,3*(actuallevel-1),MPI_INT,neigh[0],tag,comm,status1);
					delete status1;
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" received from processor " << neigh[0] << " data: "<< localarrayrecv[0] <<","<< localarrayrecv[1] <<","<< localarrayrecv[2] << endl;
					
					int pp;
					for(pp=0;pp<3*(actuallevel-1);pp=pp+3){
						localarraypert[0]=localarrayrecv[pp];
						localarraypert[1]=localarrayrecv[pp+1];
						localarraypert[2]=localarrayrecv[pp+2];
						if(localarraypert[0]!=-1 && localarraypert[1]!=-1 && localarraypert[2]!=-1){
							perturb_realization(tem,realization,localarraypert,hashrealization,my_id,iter);
							if(DEBUG)cout << "(" << iter << "+"<<actuallevel<<")SLAVE " << my_id <<" perturbing data from parent "<< neigh[0] <<": " << localarraypert[0] << "," << localarraypert[1] << "," << localarraypert[2] << endl;
						}
					}
				localarraysend=random_block(realization,my_id,iter+actuallevel-1,rand1,rand2,rand3);
				
				if(DEBUG)cout << "(" << iter << "+"<<actuallevel<<")SLAVE " << my_id <<" perturbing data: " << localarraysend[0] << "," << localarraysend[1] << "," << localarraysend[2] << endl;


				if(*numneigh==1){//leaf node
					double *old_value=new double;
					*old_value=evaluate_realization(hash,hashrealization,prom);
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" calculate old value " << *old_value << endl;
					perturb_realization(tem,realization,localarraysend,hashrealization,my_id,iter);
					double *new_value=new double;
					*new_value=evaluate_realization(hash,hashrealization,prom);
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" calculate new value " << *new_value << endl;
				

					int *decision=new int;
					*decision=decide_perturbation(*old_value, *new_value,T,my_id,iter+actuallevel-1,num_procs,randarray2 );
					if(*decision){
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" accept perturbation, end of the tree" << endl;
					}
					else{
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" reject perturbation, end of the tree" << endl;
						perturb_realization(tem,realization,localarraysend,hashrealization,my_id,iter);
					}
					
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" waiting for receive signal from processor " << neigh[0] << endl;
					
					MPI_Status *status2=new MPI_Status;
					MPI_Recv(signal,1,MPI_INT,neigh[0],tag,comm,status2);
					delete status2;
					string *state=new string;
					*state = *signal==1?"GO":"STOP";
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" received signal "<< *state <<" from processor " << neigh[0] << endl;
					delete state;
					if(*signal){
						int *info=new int[4];
						info[0]=localarraysend[0];
						info[1]=localarraysend[1];
						info[2]=localarraysend[2];
						info[3]=*decision;
					

						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" sending info: "<< info[0] << "," << info[1] << "," << info[2] << "," << info[3] << " to processor " << neigh[0] << endl;
						MPI_Send(info,4,MPI_INT,neigh[0],tag,comm);
						delete [] info;

					}
					if(*decision){/*return to original state before localarraysend perturbation*/
						perturb_realization(tem,realization,localarraysend,hashrealization,my_id,iter);
					}
					
					int ppp;
					for(ppp=0;ppp<3*(actuallevel-1);ppp=ppp+3){
						localarraypert[0]=localarrayrecv[ppp];
						localarraypert[1]=localarrayrecv[ppp+1];
						localarraypert[2]=localarrayrecv[ppp+2];
						if(localarraypert[0]!=-1 && localarraypert[1]!=-1 && localarraypert[2]!=-1){
							perturb_realization(tem,realization,localarraypert,hashrealization,my_id,iter);
						}
					}
					delete decision;
					delete old_value;
					delete new_value;
				}
				else{//internal node
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" sending data: "<< localarraysend[0] <<","<< localarraysend[1] <<","<< localarraysend[2] << " to processor " << neigh[1] << endl;

					int *arrayaccept=new int[3*(actuallevel)];
					int *arrayreject=new int[3*(actuallevel)];
					int q;
					for(q=0;q<3*(actuallevel-1);q++){
						arrayaccept[q]=localarrayrecv[q];
						arrayreject[q]=localarrayrecv[q];
					}
					arrayaccept[q]=localarraysend[0];
					arrayreject[q]=-1;
					q++;
					arrayaccept[q]=localarraysend[1];
					arrayreject[q]=-1;
					q++;
					arrayaccept[q]=localarraysend[2];
					arrayreject[q]=-1;
					MPI_Send(arrayaccept,3*(actuallevel),MPI_INT,neigh[1],tag,comm);
					MPI_Send(arrayreject,3*(actuallevel),MPI_INT,neigh[2],tag,comm);

					delete [] arrayaccept;
					delete [] arrayreject;
					

					double *old_value=new double;
					*old_value=evaluate_realization(hash,hashrealization,prom);
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" calculate old value " << *old_value << endl;
					perturb_realization(tem,realization,localarraysend,hashrealization,my_id,iter);
					double *new_value=new double;
					*new_value=evaluate_realization(hash,hashrealization,prom);
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" calculate new value " << *new_value << endl;
				

					int *decision=new int;
					*decision=decide_perturbation(*old_value, *new_value,T,my_id,iter+actuallevel-1,num_procs,randarray2 );
					if(*decision){
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" accept perturbation, internal node" << endl;
						*accepter=neigh[1];
						*rejector=neigh[2];
					}
					else{
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" reject perturbation, internal node" << endl;
						perturb_realization(tem,realization,localarraysend,hashrealization,my_id,iter);
						*accepter=neigh[2];
						*rejector=neigh[1];
					}

					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" waiting for receive signal from processor " << neigh[0] << endl;
					MPI_Status *status3=new MPI_Status;
					MPI_Recv(signal,1,MPI_INT,neigh[0],tag,comm,status3);
					delete status3;
					string *state=new string;
					*state = *signal==1?"GO":"STOP";
					if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" received signal "<< *state <<" from processor " << neigh[0] << endl;
					delete state;
					if(*signal){
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" sending signal GO to processor "<<*accepter<<" and signal STOP to processor "<<*rejector << endl;
						int *go=new int;
						int *stop=new int;
						go[0]=1;
						stop[0]=0;
						MPI_Send(go,1,MPI_INT,*accepter,tag,comm);
						MPI_Send(stop,1,MPI_INT,*rejector,tag,comm);
						delete go;
						delete stop;
						
						int *level=new int;
						int *counter=new int;
						int *i=new int;
						*level=0;
						*counter=0;
						*i=1;
						while(my_id>*counter){
							*counter=*counter+(int)pow(2,*i);
							(*i)++;
							(*level)++;
						}
						if(DEBUG)cout << "("<< iter <<"+"<<actuallevel<<")SLAVE " << my_id << " in level " << *level << endl;

						int *remaining_levels=new int;
						*remaining_levels=num_levels-(*level+1);
			
						int *infoleaf=new int[4*(*remaining_levels)];
						int *info=new int[4+4*(*remaining_levels)];
	
						info[0]=localarraysend[0];
						info[1]=localarraysend[1];
						info[2]=localarraysend[2];
						info[3]=*decision;
			
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" waiting for receive info from processor " << *accepter << endl;
						MPI_Status *status4=new MPI_Status;
						MPI_Recv(infoleaf,4*(*remaining_levels),MPI_INT,*accepter,tag,comm,status4);
						delete status4;
						if(DEBUG){
							cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" received info: ";
							for(*i=0;*i<4*(*remaining_levels);(*i)++){
								if(*i==4*(*remaining_levels)-1)
									cout << infoleaf[*i] << " from processor " << *accepter << endl;
								else
									cout << infoleaf[*i] << ",";
							}
						}
			
						for(*i=0;*i<4*(*remaining_levels);(*i)++)
							info[4+*i]=infoleaf[*i];
				
						if(DEBUG){
							cout <<"("<<iter<<"+"<<actuallevel<<")"<<"SLAVE "<< my_id <<" sending info: ";
							for(*i=0;*i<4*(*remaining_levels)+4;(*i)++){
								if(*i==4*(*remaining_levels)+3)
									cout << info[*i] << " to processor " << neigh[0] << endl;
								else
									cout << info[*i] << ",";
							}
						}
						MPI_Send(info,4*(*remaining_levels)+4,MPI_INT,neigh[0],tag,comm);
						delete [] infoleaf;
						delete [] info;
						delete i;
						delete level;
						delete counter;
						delete remaining_levels;
					}
					else{/*signal[0]==0*/
						if(DEBUG)cout <<"("<<iter<<"+"<<actuallevel<<")"<< "SLAVE "<< my_id <<" sending signal STOP to processor "<<*accepter<<" and signal STOP to processor "<<*rejector << endl;
						int *stop=new int;
						stop[0]=0;
						MPI_Send(stop,1,MPI_INT,*accepter,tag,comm);
						MPI_Send(stop,1,MPI_INT,*rejector,tag,comm);
						delete stop;
					}
					
					if(decision[0]){/*return to original state before localarraysend perturbation*/
						perturb_realization(tem,realization,localarraysend,hashrealization,my_id,iter);
					}

					int ppp;
					for(ppp=0;ppp<3*(actuallevel-1);ppp=ppp+3){
						localarraypert[0]=localarrayrecv[ppp];
						localarraypert[1]=localarrayrecv[ppp+1];
						localarraypert[2]=localarrayrecv[ppp+2];
						if(localarraypert[0]!=-1 && localarraypert[1]!=-1 && localarraypert[2]!=-1){
							perturb_realization(tem,realization,localarraypert,hashrealization,my_id,iter);
						}
					}

					delete decision;
					delete new_value;
					delete old_value;
				}
				delete signal;
				delete [] localarraysend;
				delete [] localarrayrecv;
				delete [] localarraypert;
				delete [] nomove;
				delete [] neigh;
				delete numneigh;
				delete accepter;
				delete rejector;
			}	

			/*updating perturbations in non-master nodes*/	
			if(num_procs>1){
				MPI_Bcast(perturbations,4*num_levels,MPI_INT,master,comm);
					int *coord1=new int[3];
					int *i,*k;
					i=new int;
					k=new int;
					for(*i=0;*i<num_levels;(*i)++){
						*k=4*(*i+1)-1;
						if(DEBUG_OUT){
							if(my_id==master)
								cout << "("<<iter+((*k-3)/4)<<") " << perturbations[*k-3] << ", " << perturbations[*k-2] << ", " << perturbations[*k-1] << "-> " << perturbations[*k] << endl;
						}
						if(perturbations[*k]>0){
							coord1[0]=perturbations[*k-3];
							coord1[1]=perturbations[*k-2];
							coord1[2]=perturbations[*k-1];
							perturb_realization(tem,realization,coord1,hashrealization,my_id,iter);
							if(DEBUG)cout <<"("<<iter<<")"<<"Process "<< my_id <<" updated data in  "<< coord1[0] <<","<< coord1[1] <<","<< coord1[2] << " from level " << *i << endl;
						}
					}
					delete [] coord1;
					delete i;
					delete k;
			}
			else{
				if(DEBUG_OUT)
					cout << "("<<iter<<") " << perturbations[0] << ", " << perturbations[1] << ", " << perturbations[2] << "-> " << perturbations[3] << endl;
			}
			delete [] perturbations;
			
			iter = iter + num_levels;
	}

	if(DEBUG_END)cout << "(END)PROCESS "<< my_id << " has finished the iterations" << endl;

	if(my_id==master){
		if(DEBUG_END)cout << "(END)MASTER is printing to a file" << endl;
		print_reservoir(realization,outfile);
	}

	if(num_procs>1){
		MPI_Comm_free(&comm);
	}

	if(dims[0]>0 && dims[1]>0){
		delete [] index;
		delete [] edges;
	}
	delete [] dims;

	delete [] rand1;
	delete [] rand2;
	delete [] rand3;
	delete [] randarray2;

	free_reservoir(realization);
	free_template(tem);
	MPI_Finalize();
	
	long end=time(NULL);
	cout << "END: " << end << endl;
	cout << "TOTAL TIME: " << (end-begin) << " secs." << endl;	

	return 0;
}


/**
 * \brief 	Realiza la actualizacion del schedule, actualizando la temperatura, el numero de intentos, el numero de repeticiones y el factor de reduccion lambda.
 * \param 	iter 			Numero de iteracion
 * \param 	irepo			Revisar main().
 * \param 	startobj 		Valor Inicial de la Funcion Objetivo
 * \param 	latestobj 		Ultimo valor de la Funcion Objetivo.
 * \param 	attempt 		Numero de veces que se ha reportado un movimiento hacia un estado con Funcion Objetivo mayor.
 * \param 	maxatt 			Revisar main().
 * \param 	T 			Temperatura.
 * \param 	lambda 			Factor de reduccion.
 * \param 	tred 			Numero de veces que se ha alcanzado el valor maxrepetitions. Si se alcanza ntemp veces, la simulacion se detiene.
 * \param 	ntemp			Revisar main().
 * \param 	maxrepetitions		Numero maximo de repeticiones en el valor de la funcion objetivo. Cuando se alcanza el maximo se reduce la 
 *					temperatura.
 * \param 	repcounter		Numero de repeticiones.
 * \param 	hash			Map con histograma de Imagen de Entrenamiento.
 * \param 	hashrealization		Map con histograma de Realizacion.
 * \param 	realization		Grilla con datos de la realizacion.
 * \param 	outfile			Archivo donde se guarda la grilla asociada a la realizacion, si se ha detectado una mejora en la funcion objetivo. 
 *					Solo se utiliza para generar una animacion de la evolucion de la simulacion.
 * \param 	file			Stream donde se guarda el histograma de la realizacion.
 * \param 	logfilename		Archivo donde se guarda el histograma de la realizacion.
 * \param 	prom			Suma de las frecuencias inversas encontradas en la imagen de entrenamiento. Se utiliza cuando la funcion 
 *					objetivo necesita calcular pesos.
 *
 *
 * Esta funcion realiza la actualizacion del schedule. Cuando se utilizan varios procesos, solamente el proceso 0 tiene acceso a esta funcion. Los demas
 * procesos utilizan load_schedule(). El valor irepo representa las iteraciones donde se realizara un reporte, es decir, una llamada a esta funcion. Si se
 * desean comparar los tiempos de ejecucion con \f$N\f$ y \f$M\f$ procesos, la variable irepo debe ser divisible por \f$log_2(N+1)\f$ y \f$log_2(M+1)\f$ y lo mas cercana posible 
 * al valor original ingresado por el usuario. Por ejemplo, para 1,3 y 7 procesos, con irepo=10000 ingresado por el usuario, se escoje irepo=9996. De 
 * esa manera se tiene un irepo igual para distintos numeros de procesos, y es posible comparar tiempos de ejecucion. 
 *
 */
void schedule_update(int iter, int irepo, double startobj, double* latestobj, int* attempt, int maxatt, double* T, double lambda,int* tred, int ntemp,int maxrepetitions, int *repcounter,map<string,int>& hash,map<string,int>& hashrealization, geometry *realization, char *outfile,ofstream &file, char *logfilename, float prom){
	double obj=evaluate_realization(hash,hashrealization,prom);
	if(((int)(iter/irepo))*irepo==iter){
		cout << "time: "<<time(NULL)<<", iter: " << iter << ", obj: " << obj << ", %: " << obj/startobj*100 << ", latestobj: "<<*latestobj<<", obj-latestobj: "<<obj-*latestobj<<",attempt: " << *attempt << ", tred: " << *tred << ", repcounter: "<<*repcounter<<", T: " << *T <<endl;
		if(*latestobj<obj){
			(*attempt)++;
		}
		else{
			print_reservoir(realization,outfile);
			
			open_logfile(file,logfilename);
			print_histogram(file,hashrealization);
			close_logfile(file);

			/*char *buff=new char[100];
			sprintf(buff,"perl genPlotter.pl %s %d",outfile,iter);
			cout << buff << endl;
			system(buff);
			system("echo \"plotter.par\" | ./pixelplt");
			delete [] buff;
*/
			if(abs((long)(*latestobj-obj))<=1.0e-7){
				(*repcounter)++;
				if(*repcounter>=maxrepetitions){
					*T=(*T)*lambda;
					(*tred)++;
					*repcounter=0;
				}
			}
			else{
				*repcounter=0;
			}
			
			*latestobj=obj;
			*attempt=0;
		}

		if(*attempt>=maxatt){
			*T=(*T)*lambda;
			(*tred)++;
			cout << "T: " << *T << endl;
			*attempt=0;
			*repcounter=0;
		}
	}
}

/**
 * \brief Imprime el schedule actual.
 * \param latestobj	Revisar schedule_update().
 * \param attempt	Revisar schedule_update().
 * \param T		Revisar schedule_update().
 * \param tred		Revisar schedule_update().
 * \param filename	Archivo donde se guardara el schedule.
 */

void print_schedule(double latestobj, int attempt, double T, int tred, char * filename){
        int i,j,k;
        ofstream printStream;
        printStream.open(filename,ios::out);
        printStream << latestobj << endl;
        printStream << attempt << endl;
        printStream << T << endl;
        printStream << tred << endl;
        printStream.close();
}

/**
 * \brief Carga el schedule desde un archivo, guardando los datos en variables.
 * \param latestobj	Revisar schedule_update().
 * \param attempt	Revisar schedule_update().
 * \param T		Revisar schedule_update().
 * \param tred		Revisar schedule_update().
 * \param filename	Archivo que guarda el schedule asociado.
 */
void load_schedule(double *latestobj, int *attempt, double *T, int *tred, char *filename){
        char line[20];
        ifstream inputStream;
        inputStream.open(filename,ios::in);
        if(!inputStream){
                cerr << "Error opening input stream" << endl;
        }
        if(inputStream.getline(line,20)){
                *latestobj=atof(line);
        }
        if(inputStream.getline(line,20)){
                *attempt=atoi(line);
        }
        if(inputStream.getline(line,20)){
                *T=atof(line);
        }
        if(inputStream.getline(line,20)){
                *tred=atoi(line);
        }
        inputStream.close();
}


/**
 * \brief Calcula el nivel dentro del arbol binario de un proceso.
 * \param my_id		Id del proceso. 
 *
 *
 * El proceso 0 esta en el nivel 1. Los procesos 1 y 2 estan en el nivel 2. Los procesos 3, 4, 5 y 6 estan en el nivel 3.
 * Asi sucesivamente.
 */
int level(int my_id){
	int lev=0,counter=0,i=1;
	while(my_id>counter){
		counter=counter+(int)pow(2,i);
		i++;
		lev++;
	}
	return lev+1;
}


/**
 * \brief Calcula el largo en filas de un archivo.
 * \param filename
 */
long filesize(char *filename){
	long begin,end;
	ifstream myfile (filename);
	begin = myfile.tellg();
	myfile.seekg (0, ios::end);
	end = myfile.tellg();
	myfile.close();
	return end-begin;
}

/**
 * \brief Carga las dimensiones del arbol binario que define la topologia.
 * \param filename
 *
 *
 * Revisar load_topology().
 */
int *load_topology_dims(char *filename){
	int *dims=new int[2];
	char line[16];
	ifstream inputStream;
	inputStream.open(filename,ios::in);
	if(!inputStream){
		cerr << "Error opening input stream" << endl;
	}
	if(inputStream.getline(line,16)){
		dims[0]=atoi(line);
	}
	if(inputStream.getline(line,16)){
		dims[1]=atoi(line);
	}
	inputStream.close();
	return dims;
}

/**
 * \brief Carga la topologia del arbol binario.
 * \param filename 	Archivo que contiene la especificacion de la topologia.
 * \param index		Arreglo con el numero de vecinos de cada nodo, de forma acumulativa.
 * \param edges		Arreglo con los vecinos de los nodos.
 *
 *
 * El formato del archivo que contiene la especificacion de la topologia es de la siguiente forma:
 * \code
 * 3
 * 4
 * 2  //numero de vecinos del nodo 0
 * 3  //numero de vecinos de los nodos 0 y 1
 * 4  //numero de vecinos de los nodos 0,1 y 2
 * 1  //primer vecino del nodo 0
 * 2  //segundo vecino del nodo 0
 * 0  //primer vecino del nodo 1
 * 0  //primer vecino del nodo 2
 * \endcode
 * Este ejemplo corresponde a la topologia de 3 procesos. El nodo 0 esta conectado con los nodos 1 y 2 y a su vez, los nodos 1 y 2 estan conectados con
 * el nodo 0.
 * La primera linea indica el largo del arreglo index. En este caso 3. La segunda linea indica el largo del arreglo edges.
 * Desde la linea 3 hasta la linea 3+(index-1), se indica el numero de vecinos. Desde la linea 3+index hasta la linea 3+index+(edges-1), 
 * se indican los vecinos de cada nodo.
 * Para ver mas ejemplos, revisar el directorio resources.
 */
void load_topology(char *filename, int *index, int *edges){
	char line[16];
	int indexsize=0,edgessize=0,counter=0;
	ifstream inputStream;
	inputStream.open(filename,ios::in);
	if( !inputStream ) {
		cerr << "Error opening input stream" << endl;
		return;
	}
	while(inputStream.getline(line,16)){
		if(counter==0){
			indexsize=atoi(line);
			counter++;
		}
		else if(counter==1){
			edgessize=atoi(line);
			counter++;
		}
		else{
			if(counter<indexsize+2){
				index[counter-2]=atoi(line);
			}
			if(counter>=indexsize+2){
				edges[counter-indexsize-2]=atoi(line);
			}
			counter++;
		}
	}
	inputStream.close();
}

/**
 * \brief Cargar camino aleatorio de numeros entre 0 y 1. Se utiliza para realizar una simulacion con un camino predeterminado.
 * \param filename	Archivo donde estan guardados los numeros aleatorios.
 * \param randomarray	Arreglo donde se almacenaran en memoria.
 */
void load_random(char *filename, float *randomarray){
	char line[30];
	int i=0;
	ifstream inputStream;
	inputStream.open(filename,ios::in);
	if( !inputStream ) {
		cerr << "Error opening input stream" << endl;
		return;
	}
	while(inputStream.getline(line,30)){
		randomarray[i]=atof(line);
		//cout << "load_random: randomarray["<< i << "]=" << randomarray[i] << endl;
		i++;
	}
	inputStream.close();
}

/**
 * \brief Cargar camino aleatorio de puntos 3D en la grilla de datos. Se utiliza para realizar una simulacion con un camino predeterminado.
 * \param filename	Archivo donde estan guardados los puntos aleatorios de la grilla.
 * \param rand1		Arreglo donde se almacenaran en memoria las primeras coordenadas.
 * \param rand2		Arreglo donde se almacenaran en memoria las primeras coordenadas.
 * \param rand3		Arreglo donde se almacenaran en memoria las primeras coordenadas.
 */
void load_random_3cols(char *filename, int *rand1, int* rand2,int* rand3){
	char line[150];
	int i=0;
	ifstream inputStream;
	inputStream.open(filename,ios::in);
	if( !inputStream ) {
		cerr << "Error opening input stream" << endl;
		return;
	}
	while(inputStream.getline(line,150)){
		//randomarray[i]=atof(line);
		sscanf(line,"%d\t%d\t%d",&rand1[i],&rand2[i],&rand3[i]);
		//cout << 
		i++;
	}
	inputStream.close();
}

/**
 * \brief Cargar template con dimensiones entregadas como parametros.
 * \param tem
 */
void load_template(geometry *tem){
	int i,j,k;
	tem->node=new block**[tem->lengthx];
    	for(i=0;i<tem->lengthx;i++){
            	tem->node[i]=new block*[tem->lengthy];
		for(j=0;j<tem->lengthy;j++){
			tem->node[i][j]=new block[tem->lengthz];
                    	for(k=0;k<tem->lengthz;k++){
                    		tem->node[i][j][k].x=i;
                    		tem->node[i][j][k].y=j;
                    		tem->node[i][j][k].z=k;
				tem->node[i][j][k].data=RELEVANT;
                    	}
            	}
    	}
}


/**
 * \brief Resetear posiciones del template.
 * \param tem
 */
void reset_template(geometry* tem){
	int i,j,k;
    	for(i=0;i<tem->lengthx;i++){
            	for(j=0;j<tem->lengthy;j++){
                    	for(k=0;k<tem->lengthz;k++){
                    		tem->node[i][j][k].x=i;
                    		tem->node[i][j][k].y=j;
                    		tem->node[i][j][k].z=k;
                    	}
            	}
    	}
}
/**
 * \brief Liberar la memoria utilizada para almacenar el template.
 * \param tem
 */
void free_template(geometry* tem){
    int i,j;
	for(i=0;i<tem->lengthx;i++){
            for(j=0;j<tem->lengthy;j++){
                    delete [] tem->node[i][j];
            }
    }
    for(i=0;i<tem->lengthx;i++){
    	delete [] tem->node[i];
    }
    delete tem;
}


/**
 * \brief Cargar la grilla con los datos con dimensiones entregadas como parametros en una estructura geometry.
 * \param reservoir	Grilla 3D
 * \param tem		Template utilizado
 * \param filename 	Nombre del archivo con los datos de la grilla
 * \param proportion	Umbral para los datos.  Numero entre 0 y 1, si data<proportion entonces WHITE, de lo contrario BLACK
 *
 *
 * El formato del archivo de entrada debe ser igual al utilizado por pixelplt (ver GSLIB) para leer datos, pero sin las primeras lineas donde se escribe el titulo,
 * y algunos datos adicionales. Solo deben ir los datos, en una sola columna.
 */

void load_reservoir(geometry *reservoir,geometry *tem,char *filename,double proportion){
	int i,j;
	reservoir->node=new block**[reservoir->lengthx];
	for(i=0;i<reservoir->lengthx;i++){
		reservoir->node[i]=new block*[reservoir->lengthy];
		for(j=0;j<reservoir->lengthy;j++){
			reservoir->node[i][j]=new block[reservoir->lengthz];
		}
	}
	char line[20];
	int x=0,y=0,z=0,counter=0;
	double data=0.0;
	ifstream inputStream;
	inputStream.open(filename,ios::in);
 	if( !inputStream ) {
   		cerr << "Error opening input stream" << endl;
   		return;
 	}              
	while(inputStream.getline(line,20)){
		if(x==reservoir->lengthx){
			x=0;
			y++;
		}
		if(y==reservoir->lengthy){
			x=0;
			y=0;
			z++;
		}
		data=atoi(line);
		reservoir->node[x][y][z].data=data<proportion?WHITE:BLACK;
		reservoir->node[x][y][z].x=x;
		reservoir->node[x][y][z].y=y;
		reservoir->node[x][y][z].z=z;
		//cout << "line: "<<line<<", data: " << data << ", store: " << reservoir->node[x][y][z].data << endl;
		//reservoir->node[x][y][z].passed=NOTPASSED;
		//reservoir->node[x][y][z].code=new string[tem->lengthx * tem->lengthy * tem->lengthz];
		counter++;
		x++;
	}
	inputStream.close();
}

/**
 * \brief Copiar una grilla almacenada en una estructura geometry.
 * \param dest
 * \param orig
 */

void copy_reservoir(geometry *dest,geometry *orig){
	int i,j,k;
	dest->node=new block**[dest->lengthx];
	for(i=0;i<dest->lengthx;i++){
		dest->node[i]=new block*[dest->lengthy];
		for(j=0;j<dest->lengthy;j++){
			dest->node[i][j]=new block[dest->lengthz];
			for(k=0;k<dest->lengthz;k++){
				dest->node[i][j][k].x=orig->node[i][j][k].x;
				dest->node[i][j][k].y=orig->node[i][j][k].y;
				dest->node[i][j][k].z=orig->node[i][j][k].z;
				dest->node[i][j][k].data=orig->node[i][j][k].data;
			}
		}
	}
}

/**
 * \brief Imprimir una grilla en un archivo.
 * \param reservoir	Grilla 3D
 * \param filename	Stream del archivo donde se guardara la grilla
 *
 *
 * El formato del archivo de salida es compatible con el programa pixelplt, perteneciente a GSLIB.
 */
void print_reservoir(geometry *reservoir,char * filename){
    	int i,j,k;
	ofstream printStream;
	printStream.open(filename,ios::out);
	printStream << "FILTERED  to " << reservoir->lengthx << " by " << reservoir->lengthy << endl;
	printStream << "1" << endl;
	printStream << "sumones" << endl;
	for(j=0;j<reservoir->lengthy;j++){
            	for(i=0;i<reservoir->lengthx;i++){
            		for(k=0;k<reservoir->lengthz;k++){
            			if((reservoir->node[i][j][k].data)==BLACK)
					printStream << 1 << endl;
				else
					printStream << 0 << endl;
			}
            	}
    	}
	printStream.close();
}


/**
 * \brief Generar grilla aleatoria.
 * \param reservoir
 * \param proportion
 */
void generate_reservoir(geometry *reservoir, double proportion){
    	int i,j,k;
	srand(time(NULL));
	for(i=0;i<reservoir->lengthx;i++){
            	for(j=0;j<reservoir->lengthy;j++){
            		for(k=0;k<reservoir->lengthz;k++){
            			reservoir->node[i][j][k].data=((double)(rand()%100 +1))/100.0<=proportion?BLACK:WHITE;
			}
            	}
    	}
}

/**
 * \brief No se utliza.
 */

void free_reservoir_nodes(geometry* reservoir){
    	int i,j;
	for(i=0;i<reservoir->lengthx;i++){
            	for(j=0;j<reservoir->lengthy;j++){
            		//for(k=0;k<reservoir->lengthz;k++){
            		//	printf("liberando %d %d %d code\n",i,j,k);
			//	free(reservoir->node[i][j][k].code);
            		//}
			free(reservoir->node[i][j]);
            	}
    	}
    	for(i=0;i<reservoir->lengthx;i++){
    		free(reservoir->node[i]);
    	}
}

/**
 * \brief Liberar memoria utilizada para almacenar la grilla.
 * \param reservoir
 */
void free_reservoir(geometry *reservoir){
    	int i,j;
	for(i=0;i<reservoir->lengthx;i++){
            	for(j=0;j<reservoir->lengthy;j++){
            		//for(k=0;k<reservoir->lengthz;k++){
            		//	printf("liberando %d %d %d code\n",i,j,k);
			//	free(reservoir->node[i][j][k].code);
            		//}
			delete [] reservoir->node[i][j];
            	}
    	}
    	for(i=0;i<reservoir->lengthx;i++){
    		delete [] reservoir->node[i];
    	}
    	delete reservoir;
}


/**
 * \brief Cargar patrones asociados a un template, desde una grilla hacia un map, guardando las frecuencias de aparicion.
 * \param tem		Template utilizado
 * \param reservoir	Grilla 3D
 * \param HASH_SIZE	Tamano maximo del map
 * \param hash		En esta estructura se almacenan las frecuencias detectadas.
 * \param file		Stream del archivo donde se guardaran las estadisticas (respaldo)
 *
 *
 * La manera en que se recorre la grilla corresponde los puntos que satisfacen
 * \f$i=0\,(\textrm{mod } tem.size.x)\f$
 * \f$j=0\,(\textrm{mod } tem.size.y)\f$
 * \f$k=0\,(\textrm{mod } tem.size.z)\f$
 * Para cada uno de esos puntos, se utiliza la funcion evaluate_template() para revisar todos los patrones asociados que utilizan
 * como soporte al punto. Se deben resetear las posiciones de la ventana para cada nuevo punto, para ello se utiliza la funcion reset_template().
 */
void load_associatedPatterns(geometry *tem, geometry *reservoir, unsigned int HASH_SIZE,map<string,int>& hash, ofstream& file){
	int i,j,k/*,m*/;
	int resx, resy, resz;
	int con=0;
	//int orig_lengthx=tem->lengthx;
	//int orig_lengthy=tem->lengthy;
	//int orig_lengthz=tem->lengthz;

	//for(m=1;m<=orig_lengthx;m++){
	//	tem->lengthx=m;
	//	tem->lengthy=m;
	//	tem->lengthz=1;
		for(resx=0;resx<reservoir->lengthx;resx=resx+tem->lengthx){
			for(resy=0;resy<reservoir->lengthy;resy=resy+tem->lengthy){
				for(resz=0;resz<reservoir->lengthz;resz=resz+tem->lengthz){
					int numPattern=0;
					//cout << reservoir->node[resx][resy][resz].data << endl;
					for(i=0;i<tem->lengthx;i++){
						for(j=0;j<tem->lengthy;j++){
							for(k=0;k<tem->lengthz;k++){
								reset_template(tem);
								evaluate_template(i,j,k,reservoir->node[resx][resy][resz],numPattern,tem,reservoir,HASH_SIZE, hash,file);
								numPattern++;
								con++;
							}
						}
					}
				}
			}
		}
	//}


	if(hash.size()>0){
		map<string, int>::const_iterator iter2;
		for (iter2=hash.begin(); iter2 != hash.end(); ++iter2) {
			file << iter2->first << " " << iter2->second << endl;
		}
	}
}

/**
 * \brief Evaluar los patrones que aparecen asociados a un punto en la grilla, para un template dado, guardando las frecuencias en un map.
 * \param despx 	Desplazamiento del template en el eje x con respecto al punto soporte b
 * \param despy		Desplazamiento del template en el eje x con respecto al punto soporte b	
 * \param despz		Desplazamiento del template en el eje x con respecto al punto soporte b 
 * \param b		Punto soporte en la grilla.
 * \param numPattern	Variable de uso interno
 * \param tem		Template utilizado
 * \param reservoir	Grilla 3D
 * \param HASH_SIZE	Tamano maximo que el map puede alcanzar. Uso interno
 * \param hash		En esta estructura se almacenan y actualizan las frecuencias.
 * \param file		Stream del archivo donde se respaldan las estadisticas en caso que se alcance el tamano maximo del map. Uso interno
 */

void evaluate_template(int despx, int despy, int despz, block b, int numPattern,geometry* tem, geometry* reservoir, unsigned int HASH_SIZE,map<string,int>& hash, ofstream& file){
	int i,j,k,counter=0,flag=0;
	int resx=b.x, resy=b.y, resz=b.z;
	string code="";
	//char chunk[10];
	for(i=0;i<tem->lengthx;i++){
		for(j=0;j<tem->lengthy;j++){
			for(k=0;k<tem->lengthz;k++){
				//sprintf(chunk,"#%d#%d#%d#",i,j,k);
				//code.append(chunk);
				tem->node[i][j][k].x = tem->node[i][j][k].x + resx - despx;
				tem->node[i][j][k].y = tem->node[i][j][k].y + resy - despy;
				tem->node[i][j][k].z = tem->node[i][j][k].z + resz - despz;
				if((tem->node[i][j][k].x==resx) && (tem->node[i][j][k].y==resy) && (tem->node[i][j][k].z==resz)){
					code.append(1,b.data==WHITE?'W':'B');//=b->code[numPattern]+"P";
					counter++;
				}
				else{
					if(tem->node[i][j][k].x<0 || tem->node[i][j][k].x>=reservoir->lengthx || tem->node[i][j][k].y<0 || tem->node[i][j][k].y>=reservoir->lengthy || tem->node[i][j][k].z<0 || tem->node[i][j][k].z>=reservoir->lengthz){
						code.append("X");//=b->code[numPattern]+"X";
						counter++;
						flag++;
					}
					else{
						if(reservoir->node[tem->node[i][j][k].x][tem->node[i][j][k].y][tem->node[i][j][k].z].data==WHITE){
							code.append("W");//[counter]='W';
							counter++;
						}
						else{
							code.append("B");//[counter]='B';
							counter++;
						}

					}
				}
			}
		}
	}

	if(flag==0){
		map<string,int>::iterator iter = hash.find(code);
		if (iter==hash.end()) {
			if(hash.size()==HASH_SIZE ){
				map<string, int>::const_iterator iter2;
				for (iter2=hash.begin(); iter2 != hash.end(); ++iter2) {
					file << iter2->first << "," << iter2->second << endl;
				}
				hash.clear();
			}
			hash.insert(make_pair(code,1));
		}
		else{
			hash[code]++;
		}
	}
}

/**
 * \brief Imprimir el histograma de frecuencias almacenado en un map en un archivo.
 * \param file
 * \param hash
 */
void print_histogram(ofstream& file,map<string,int>& hash){
	map<string, int>::const_iterator iter2;
	for (iter2=hash.begin(); iter2 != hash.end(); ++iter2) {
		file << iter2->first << "," << iter2->second << endl;
	}
}
/**
 * \brief Funcion analoga a evaluate_template() pero que retorna un string. 
 * 
 *
 * El string retornado puede ser "OUT" u otro patron distinto. Si es "OUT", el patron
 * detectado no es ingresado en las estadisticas, pues utiliza puntos fuera de la grilla.
 */
string evaluate_template_ret(int despx, int despy, int despz, int * coord, int numPattern,geometry* tem, geometry* reservoir,map<string,int>& hash){
	int i,j,k,counter=0,flag=0;
	int resx=coord[0], resy=coord[1], resz=coord[2];
	string code="";
	//char chunk[10];
	for(i=0;i<tem->lengthx;i++){
		for(j=0;j<tem->lengthy;j++){
			for(k=0;k<tem->lengthz;k++){
				//sprintf(chunk,"#%d#%d#%d#",i,j,k);
				//code.append(chunk);
				tem->node[i][j][k].x = tem->node[i][j][k].x + resx - despx;
				tem->node[i][j][k].y = tem->node[i][j][k].y + resy - despy;
				tem->node[i][j][k].z = tem->node[i][j][k].z + resz - despz;
				if((tem->node[i][j][k].x==resx) && (tem->node[i][j][k].y==resy) && (tem->node[i][j][k].z==resz)){
					code.append(1,reservoir->node[resx][resy][resz].data==WHITE?'W':'B');//=b->code[numPattern]+"P";
					counter++;
				}
				else{
					if(tem->node[i][j][k].x<0 || tem->node[i][j][k].x>=reservoir->lengthx || tem->node[i][j][k].y<0 || tem->node[i][j][k].y>=reservoir->lengthy || tem->node[i][j][k].z<0 || tem->node[i][j][k].z>=reservoir->lengthz){
						code.append("X");//=b->code[numPattern]+"X";
						counter++;
						flag++;
					}
					else{
						if(reservoir->node[tem->node[i][j][k].x][tem->node[i][j][k].y][tem->node[i][j][k].z].data==WHITE){
							code.append("W");//[counter]='W';
							counter++;
						}
						else{
							code.append("B");//[counter]='B';
							counter++;
						}

					}
				}
			}
		}
	}
	if(flag>0)
		return "OUT";
	else
		return code;	
}

/**
 * \brief Abrir archivo log, donde quedara registrado el histograma de frecuencias en cada actualizacion.
 * \param file
 * \param logfilename
 */
void open_logfile(ofstream &file, char *logfilename){
	file.open(logfilename,ios::out);
}

/**
 * \brief Imprimir una linea en el archivo log. 
 * \param file
 * \param line
 */

void print_to_log(ofstream &file, char *line){
	file << line << endl;
}
/**
 * \brief Cerrar archivo log, donde quedara registrado el histograma de frecuencias en cada actualizacion.
 * \param file
 */

void close_logfile(ofstream &file){
	file.close();
}


/**
 * \brief Peso total de los inversos de las frecuencias. Se utiliza cuando se setea la funcion objetivo con Pesos.
 * \param hash
 * \param c
 *
 *
 * En la variable c se almacena el siguiente resultado
 * \f$\sum_{i\in\mathcal P }\frac{1 }{ f^{TI}_i} \f$
 */
float total_weight(map<string,int>& hash,float c){
	float total=0.0;
	map<string,int>::const_iterator iterHash;
	for(iterHash=hash.begin();iterHash!=hash.end();++iterHash){
		total=total+1.0/((float)(iterHash->second)+c);
	}
	return total;
}

/**
 * \brief Calcular un bloque o punto aleatorio en la grilla utilizando el camino aleatorio especificado por rand1, rand2 y rand3.
 * \param reservoir
 * \param proc
 * \param iter
 * \param rand1
 * \param rand2
 * \param rand3
 */
int *random_block(geometry* reservoir, int proc, int iter, int* rand1, int* rand2, int* rand3){
	int *coord=new int[3];
	//srand((time(NULL)*(proc+1))+iter);
	coord[0]=rand1[iter] % reservoir->lengthx;
	coord[1]=rand2[iter] % reservoir->lengthy;
	coord[2]=rand3[iter] % reservoir->lengthz;
	//coord[0]=rand() % reservoir->lengthx;
	//coord[1]=rand() % reservoir->lengthy;
	//coord[2]=rand() % reservoir->lengthz;
	return coord;
}

/**
 * \brief Evaluar la funcion objetivo en un estado.
 * \param hash			Map con histograma de frecuencias de imagen de entrenamiento
 * \param hashrealization	Map con histograma de frecuencias de realizacion
 * \param prom			Valor promedio de frecuencias inversas para imagen de entrenamiento
 *
 *
 * Actualmente se encuentra implementada la siguiente funcion objetivo:
 * \f$\frac{\frac{1 }{ f^{TI}_i}}{\sum_{j\in\mathcal P }\frac{1 }{ f^{TI}_j} }\|f^{TI}_i-f^{RE}_i\|^2\f$
 * Para ver mas detalles sobre la inclusion de una nueva funcion objetivo, revisar el trabajo de titulo, capitulo Implementacion.
 */
double evaluate_realization(map<string,int>& hash, map<string,int>& hashrealization,float prom){
	float total=0.0;
	float totalInv=0.0;


	//map<string,int> copyhashrealization (hashrealization);

	//if(hash.size()>hashrealization.size()){
		map<string,int>::const_iterator iterHash;
		map<string,int>::iterator iterHashRe;
		for(iterHash=hash.begin();iterHash!=hash.end();++iterHash){
			iterHashRe = hashrealization.find(iterHash->first);
			if (iterHashRe!=hashrealization.end()/*hashrealization.count(iterHash->first)>0*/) {
#if INVERSEWEIGHT==0
				total=total+/*((1.0/(iterHash->second+1.0))/prom)**/pow(iterHash->second - hashrealization[iterHash->first],2);
#else
				total=total+((1.0/(iterHash->second+1.0))/prom)*pow(iterHash->second - hashrealization[iterHash->first],2);
#endif
			//	hashrealization[iterHash->first]=0;
			}
			else{
#if INVERSEWEIGHT==0
				total=total+/*((1.0/(iterHash->second+1.0))/prom)**/pow(iterHash->second - 0,2);
#else
				total=total+((1.0/(iterHash->second+1.0))/prom)*pow(iterHash->second - 0,2);
#endif
			}
		}
	//}
	//else{
		map<string,int>::const_iterator iterHashRe2;
		map<string,int>::iterator iterHash2;
		for(iterHashRe2=hashrealization.begin();iterHashRe2!=hashrealization.end();++iterHashRe2){
			//total=total+((1.0/(hash[iterHashRe2->first]+1.0))/prom)*pow(hash[iterHashRe2->first]-iterHashRe2->second,2);
			iterHash2 = hash.find(iterHashRe2->first);
			if (iterHash2==hash.end()/* hash.count(iterHashRe2->first)==0*/) {
				total=total+/*(1.0/prom)**/pow(0 - iterHashRe2->second,2);
			}
		}
	//}

	return (double)total;
}

/**
 * \brief Perturbar un estado, cambiando de color un punto de la grilla, tras lo cual se actualiza el map asociado a la grilla, donde se guarda el histograma de frecuencias.
 * \param tem			Template utilizado
 * \param reservoir		Grilla 3D
 * \param coord			Coordenadas de un punto en la grilla 3D
 * \param hashrealization	Map que contiene las estadisiticas, las cuales se actualizaran
 * \param my_id			Id del proceso
 * \param iterout		Numero de iteracion
 *
 *
 * La perturbacion afecta solamente a los patrones asociados al punto perturbado. 
 * Para ello, se utilza la funcion evaluate_template() para obtener los patrones asociados, luego se revisa el map para ver si existen o no, 
 * y se reemplazan por los nuevos patrones obtenidos. Primero se resta 1 a la frecuencia de los patrones antes de la perturbacion 
 * y luego se suma 1 a la frecuencia de los patrones despues de la perturbacion.
 */
void perturb_realization(geometry *tem, geometry* reservoir,int *coord,map<string,int>& hashrealization, int my_id, int iterout){
	//int orig_lengthx=tem->lengthx, orig_lengthy=tem->lengthy, orig_lengthz=tem->lengthz;
	char dataaux;
	string procname;
	if(my_id==0){
		procname="MASTER";
	}
	else{
		procname="SLAVE";
	}

	if(reservoir->node[coord[0]][coord[1]][coord[2]].data==WHITE){
		dataaux='W';
	}
	else{
		dataaux='B';
	}
	if(DEBUG)cout << "("<<iterout<<")"<<procname<<" " << my_id << " removing patterns associated with node " << coord[0] << "," << coord[1] << "," << coord[2] << " with data " << dataaux << endl;
	int i,j,k/*,m*/;
	string code="";


	//for(m=1;m<=orig_lengthx;m++){

	//tem->lengthx=m;
	//tem->lengthy=m;
	//tem->lengthz=1;
	for(i=0;i<tem->lengthx;i++){
		for(j=0;j<tem->lengthy;j++){
			for(k=0;k<tem->lengthz;k++){
				reset_template(tem);
				code=evaluate_template_ret(i,j,k,coord,0,tem,reservoir, hashrealization);
				if(code.compare("OUT")!=0){
					map<string,int>::iterator iter = hashrealization.find(code);
					if (iter!=hashrealization.end()) {
						if(DEBUG){ 
							cout << "(" << iterout << ")" <<  procname << " " << my_id << " removing " << code << endl;
						}
						if(hashrealization[code]>0){
							hashrealization[code]--;
						}
						if(hashrealization[code]==0){
							hashrealization.erase(iter);
						}
					}
				}
			}
		}
	}
	//}

	if(reservoir->node[coord[0]][coord[1]][coord[2]].data==WHITE){
		reservoir->node[coord[0]][coord[1]][coord[2]].data=BLACK;
	}
	else{
		reservoir->node[coord[0]][coord[1]][coord[2]].data=WHITE;
	}
//	reservoir->node[coord[0]][coord[1]][coord[2]].data=reservoir->node[coord[0]][coord[1]][coord[2]].data==WHITE?BLACK:WHITE;
	
	if(reservoir->node[coord[0]][coord[1]][coord[2]].data==WHITE){
		dataaux='W';
	}
	else{
		dataaux='B';
	}

	if(DEBUG)cout << "(" << iterout << ")" << procname << " " << my_id << " adding patterns associated with node " << coord[0] << "," << coord[1] << "," << coord[2] << " with data " << dataaux << endl;

 	//for(m=1;m<=orig_lengthx;m++){

        //tem->lengthx=m;
        //tem->lengthy=m;
        //tem->lengthz=1;

	for(i=0;i<tem->lengthx;i++){
		for(j=0;j<tem->lengthy;j++){
			for(k=0;k<tem->lengthz;k++){
				reset_template(tem);
				code=evaluate_template_ret(i,j,k,coord,0,tem,reservoir, hashrealization);
				if(code.compare("OUT")!=0){
					map<string,int>::iterator iter = hashrealization.find(code);
					if(DEBUG){ 
						cout << "(" << iterout << ")" <<  procname << " " << my_id << " adding " << code << endl;
					}
					if (iter!=hashrealization.end()) {
						//if(DEBUG){ 
						//	cout << "(" << iterout << ")" <<  procname << " " << my_id << " adding " << code << endl;
						//}
						hashrealization[code]++;
					}
					else{
						hashrealization.insert(make_pair(code,1));
					}
				}
			}
		}
	}
	//}
}

/**
 * \brief Decidir si se acepta o no una perturbacion. 
 * \param old_value	Valor de la funcion objetivo antes de la perturbacion
 * \param new_value	Valor de la funcion objetivo despues de la perturbacion
 * \param temp		Temperatura
 * \param proc		Id del proceso
 * \param iter		Numero de iteracion
 * \param num_procs	Numero de procesos
 * \param randarray	Arreglo con numeros aleatorios entre 0 y 1
 *
 *
 * La regla de decision es la siguiente: 
 * \f$P(\textrm{aceptar estado }j\textrm{ desde estado }i)=\textrm{exp}\left( -\frac{O_j-O_i}{T} \right)\,\textrm{ si }O_j<O_i\f$
 * \f$P(\textrm{aceptar estado }j\textrm{ desde estado }i)=0\,\textrm{ si }\O_j\geq O_i\f$
 * Se utiliza el camino aleatorio de numeros entre 0 y 1 almacenado en randarray.
 */
int decide_perturbation(double old_value, double new_value, double temp, int proc, int iter, int num_procs, float *randarray){
	if(old_value>=new_value){
		if(DEBUG){
			cout << "("<<iter<<")DEBUG_AC old: " << old_value << ", new: " << new_value << endl;
		}
		//cout << iter << ":" << 1 << ", T="<< temp <<  endl;
		return 1;
	}
	else{
		//srand(time(NULL)*(proc+1)+iter);
		float r=randarray[iter];//((float)rand())/((float)RAND_MAX);
		//cout <<  "decide_perturbation: randarray[" << iter << "]=" <<  r  << endl;

		if(exp(-(new_value-old_value)/(temp))>r){
			if(DEBUG){
				cout << "("<<iter<<")XXXXDEBUG_AC old: " << old_value << ", new: " << new_value << ", probability:" << exp(-(new_value-old_value)/(temp)) << ", random: " << r << endl;
			}
			//cout << iter << ":" << 2 << ", T="<<temp<< ", r="<< r<< endl;
			return 2;
		}
		else{
			if(DEBUG){
				cout << "("<<iter<<")DEBUG_RE old: " << old_value << ", new: " << new_value << ", probability:" << exp(-(new_value-old_value)/(temp)) << ", random: " << r << endl;
			}
			//cout << iter << ":" << 0 << ", T="<<temp<<", r="<< r<<endl;
			return 0;
		}
	}
}
