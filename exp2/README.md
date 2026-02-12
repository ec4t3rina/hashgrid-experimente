update temporal ratings!!
* avem 25 de noduri de tip new_i (primele 5 sunt "autoritatile")
* facem primele interactiuni pretraining
* calculam embeddingsurile nodurilor (pagerank + centralitate)
* activam trainingul (o data la 3 pasi) - am lasat sa ruleze pe 30 de pasi
* filtrare: prezicem scorurile, le selectam pe cele cu rating >0.55
* polarizare graduala pt a separa clar nodurile succesful
* analizam rezultate finale pe grafic
