Legenda :
- $\texttt{m}$ : index for machines
- $\texttt{p}$ : index for products in an order
- $\texttt{c}$ : index for production cycles
- $\texttt{l}$ : index for levate in some cycle

Decision Variables ( require search )
---

ASSIGNMENTS
- $\texttt{A[p,c,m]}$
	- **type** : Boolean
	- **domain** : $\{0,1\}$
	- **description** : keeps track of (product, cycle, machine) assignments

COMPLETE / PARTIAL CYCLES
- $\texttt{COMPLETE[p,c]}$
	- **type** : Boolean
	- **domain** : ${0,1}$
	- **description** : cycle $\texttt{c}$ of $\texttt{p}$ is a complete cycle

NUMBER OF LEVATE
- $\texttt{NUM_LEVATE[p,c]}$
	-  **type** : Integer
	- **domain** : $\texttt{[1, max\_levate(p)]}$
	- **description** : number of levate 

TIME BEGINNINGS
- $\texttt{SETUP_BEG[p,c]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_domain}$
	- **description** : <u>beginning</u> of machine machine START operation
- $\texttt{LOAD_BEG[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_domain}$
	- **description** : <u>beginning</u> of machine machine LOAD operation
- $\texttt{UNLOAD_BEG[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_domain}$
	- **description** : <u>beginning</u> of machine machine UN-LOAD operation

VELOCITY
- $\texttt{VELOCITY[p,c]}$ :
	- **type** : Integer
	- **domain** : $\{-1, 0, 1\}$
	- **description** : VELOCITY at which cycle $\texttt{c}$ of product $\texttt{p}$ runs


Other Variables ( no search needed, easily calculated )
---

COSTS RELATED
- $\texttt{SETUP_COST[p,c]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine START operation
- $\texttt{LOAD_COST[p,c,l]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine LOAD operation
- $\texttt{UNLOAD_COST[p,c,l]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine UN-LOAD operation 
- $\texttt{LEVATA_COST[p,c]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine LEVATA operation 

END OF PROCEDURES
- $\texttt{SETUP_END[p,c]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>end</u> of machine machine START operation
- $\texttt{LOAD_END[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>end</u> of machine machine LOAD operation
- $\texttt{UNLOAD_END[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>end</u> of machine machine UN-LOAD operation

Constraints
---


