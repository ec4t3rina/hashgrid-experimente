 Plan of action: 
- rulam cateva ticuri, se dau scoruri random intre noduri
- calculam embeddingsurile nodurilor bazat pe aceste interactiuni (pagerank + centralitate in graf)
- pornim din nou, de data asta prezicem scorurile bazat pe noduri
- vedem ce rezultate avem (1)
- antrenam un model care sa favorizeze conexiunile cu autoritate mare in retea (are legatura cu scoruri mari/conectat cu noduri cu autoritate mare) (nu am favorizat edgeurile cu scor mare deocamdata dar poate sa fie transformat usor, varianta asta mi se parea mai usor de verificat)
- rulam + vedem distributia rezultatelor (2), comparam cu rezultatele trecute (sa vedem ca antrenamentul a facut ceva)
- functie de polarizare (ca sa separam nodurile in mai bune/mai rele)
- distributie din nou (diverse degreeuri, 3 si 4)
