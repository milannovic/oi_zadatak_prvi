1. Prvo kopirajte repozitorijum
2. Onda vrsimo kompajliranje u terminalu repozitorijuma:
   gcc -O2 -std=c11 cover_knapsack_eval_final.c -o cover_eval
3. Preko terminala pokrecemo program
   Primjer pokretanja:
   ./cover_eval 100 1000 20 12345 all 0.3 0.5 0.7 results_20_all.cs
   ./cover_eval 800 1000 5 12345 all 0.5 results.csv
   ./cover_eval 120 1000 20 12345 all 0.1 0.3 0.5 0.7 0.9 results1.csv
