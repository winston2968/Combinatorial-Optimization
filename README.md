# Algorithmique avancée - M1 info - TPs PLNE+MH - devoir 2

Le sujet du dernier TP d'Algo Avancée cette année, et du devoir à déposer sur Moodle à la fin du semestre, est un jeu de Tournoi de combats, sur lequel on vous demande de comparer les approches PLNE et métaheuristique.

Plus précisément, une équipe de **combattants** doit affronter une équipe **d'hôtes** dans une série de **combats singuliers** : chaque combat voit un combattant et un hôte s'affronter ; l'issue de chaque combat dépend des niveaux de compétences du combattant et de l'hôte : le combattant gagne si son niveau de compétences est plus élevé que celui de l'hôte ; perd si son niveau de compétences est moins élevé que celui de l'hôte ; il y a match nul si les deux niveaux de compétences sont identiques. Votre objectif est **d'organiser les combats** de manière à **maximiser les gains** de l'équipe de combattants, en respectant les règles suivantes :

- L'équipe des combattants peut désigner un **capitaine**, mais ça n'est pas obligatoire ;

- Chaque combattant peut engager **deux combats** (ou moins) - sauf le capitaine des combattants, s'il y en a un, qui ne peut engager qu'un combat (ou aucun) ;

- Chaque hôte peut être engagé dans un combat au maximum ;

- L'équipe des combattants a un budget en énergie $B$ qui ne peut être dépassé : à chaque hôte $j$ est en effet associé un coût énergétique $E_j$ pour le combattre (quelle que soit l'issu du combat), et la somme des coûts énergétique des combats engagés par les combattants ne peut dépasser $B$.

Les gains sont calculés de la manière suivante : à chaque hôte $j$ sont associés un profit $W_j \in \mathbb{N}$ ainsi qu'un coût $L_j \in \mathbb{N}$ ; si un combattant $i$ est engagé contre l'hôte $j$ et le bat, le profit $W_j$ est ajouté aux gains de l'équipe de combattants ; si au contraire $i$ est battu par $j$ alors $L_j$ est soustrait des gains de l'équipe de combattants ; rien n'est ajouté ni soustrait des gains de l'équipe des combattants lors d'un match nul. S'il y a un capitaine, celui-ci voit son niveau de compétence augmenté de $5$ pour tous ses combats. Il y a aussi une pénalité $P$ de refus de combat qui est déduite des gains de l'équipe des combattants pour chaque hôte qui n'est pas combattu.

Chaque instance de ce problème est décrit dans un fichier qui contient :

- En début de fichier, sur des lignes séparées, et dans cet ordre : le nombre $C$ de combattants ; le nombre $H$ d'hôtes ; le budget énergétique $B$ ; la pénalité $P$ déduite des gains des combattants pour chaque hôte qui n'est pas combattu ;
- Puis, sur chacune des $C$ lignes suivantes : l'identifiant d'un combattant et son niveau de compétence ;
- Enfin, sur chacune des $H$ lignes suivantes, et dans cet ordre : l'identifiant $j$ d'un hôte, son niveau de compétence, $W_j$, $L_j$, $E_j$.

(Il y a aussi des lignes de commentaires, commençant par #.)

Une **solution** est donc une liste de paires $(i,j)$ indiquant les combats sélectionnés. Elle doit satisfaire les contraintes ci-dessus, et maximiser les gains pour l'équipe des combattants.

## Joker :

Dans une version étendue, l'équipe des combattants a une unique carte Joker : cette carte peut être attribuée à l'un des combattants, qui peut l'utiliser pour un ou plusieurs de ces combats ; lorsqu'elle est utilisée par un combattant, cette carte multiplie par 2 les profits et les coûts associés aux combats de ce combattant.

## Dépôt :

On vous demande de déposer **3 fichiers sur Moodle**, avant la date limite :

- Deux fichiers avec le suffixe ```.py``` contenant deux programmes pythons, l'un étant votre modélisation en PLNE pour résoudre le problème avec ```PySCIPOpt``` et ```SCIP```, l'autre étant votre programme pour le résoudre le problème avec une métaheuristique ; 

- Chacun des fichiers devra avoir, en haut, les déclarations du nom du fichier d'entrées et des éventuels paramètres (nombre d'itérations max. par exemple ou longeur de la liste tabou en MH), afin qu'on puisse facilement tester votre programme sur différentes instance et avec différents paramètres ; chaque programme doit pouvoir être testé avec la commande ```python3 votre_fichier.py``` ou ```python votre_fichier.py```, éventuellement avec des paramètres qui devront alors être expliqués au début des fichiers ; et :
- Un **document pdf** décrivant :
    1) Quelle version du problème vous avez résolu (avec ou sans capitaine / joker) ;
    2) Votre modèle en PLNE ;
    3) Pour la partie MH, les structures de données utilisées, et les méthodes implémentées pour le calcul d'une solution initiale, l'exploration du voisinage, la recherche locale (avec ou sans liste tabou) ;
    4) Une comparaison des deux approches sur le tournoi de combats : on attend notamment un, ou plusieurs, ```tableau(x)``` indiquant, pour chaque instance traitée, la valeur de la meilleure solution trouvée en MH et en PLNE, le temps de calcul, le nombre de nœuds explorés (en PLNE) ou le nombre d'itérations (en PLNE, si la résolution ne termine pas en temps raisonnable, on pourra donner un encadrement de la valeur optimale tel qu'affiché par le solveur, en indiquant le temps qu'il a fallu pour arriver à cet encadrement).

## Indications / suggestions d'implémentations

On pourra facilement énumérer des paires avec la fonction combination de la librairie ```itertools```.

### PLNE

Ce problème ressemble, de loin, au problème du sac-à-dos, puisqu'il faut sélectionner des combats, mais avec des gains qui dépendent du statut des combattants (capitaine ou non), et des niveaux de compétences. On pourra commencer par implémenter une version sans capitaine et :

- calculer une matrice de paramètres indiquant, pour chaque combattant $i$ et chaque hôte $j$, ce que le combat $i$ contre $j$ rapporte à l'équipe de combattants ;
- définir une matrice de variables indiquant, pour chaque combattant $i$ et chaque hôte $j$, si le combat $i$ contre $j$ est sélectionné ou non ;
- définir un vecteur de variables indiquant, pour chaque hôte, s'il est combattu ou non.

### MH

Pour calculer une solution initiale, on pourra utiliser un **algorithme greedy** qui énumère les paires $(i,j)$ possibles et ajoute les combats correspondants lorsque ça ne fait pas dépasser le budget énergétique ni les contraintes sur les nombres de combats maximaux. Pour pouvoir faire des redémarrages avec des solutions initiales différentes, il faudra ajouter un élément aléatoire à cette énumération. (Par exemple, les fonctions ```shuffle``` et ```sample``` de la librairie ```random``` permettent de générer des permutations aléatoires d'une liste.) Pour le voisinage d'une solution ```S```, on pourra tester des plusieurs opérateurs, par exemple qui ajoutent, suppriment, ou échangent des combats à/de la solution courante.