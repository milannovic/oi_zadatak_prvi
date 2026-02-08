/*
  Cover Knapsack problem:
    min |S|  s.t. sum_{i in S} w_i >= W

  Racunarska evaluacija
   - Generise instance prema Kellerer familijama (tezine):
       * uncorr : w_i ~ Uniform[1, R]
       * subset : w_i ~ Uniform[1, R] (u knjizi je "subset sum" kad je p_i = w_i; ovdje profit ne koristimo)
       * simw   : w_i ~ Uniform[100000, 100100] ("uncorrelated with similar weights" tezine)
   - Demand W postavlja kao procenat ukupne tezine: W = alfa * sum(w)
   - Algoritmi:
       * Greedy: uzmi najvece tezine dok ne pokrijes W
       * Optimalno: tacno rjesenje ILP-a preko Branch-and-Bound (x_i ∈ {0,1})
   - Izvjestaj:
       * tabela po instancama i alfa
       * prosjecno vrijeme, prosjecni gap, % optimalnih greedy

  Kompajliranje:
    gcc -O2 -std=c11 cover_knapsack_eval_final.c -o cover_eval

  Primjer pokretanja:
    ./cover_eval 60 1000 30 12345 uncorr 0.3 0.5 0.7 results.csv

  Argumenti:
    argv[1] n               (broj predmeta)           default 60
    argv[2] R               (data range)              default 1000
    argv[3] trials          (broj instanci)           default 20
    argv[4] seed            (sjeme RNG)               default time()
    argv[5] family          (uncorr|subset|simw|all)  default all
    argv[6..] alphas        (npr. 0.3 0.5 0.7)        default 0.5
    last arg (opciono): csv fajl (npr. results.csv)

  Napomena:
    BnB je egzaktan, ali za velike n moze postati spor. Tipicno n<=80 radi okej.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define INF 1000000000

typedef struct {
    int w;
    int idx;
} Item;

/* Poredjenje po opadajucoj tezini */
static int cmp_desc_w(const void *a, const void *b) {
    const Item *ia = (const Item*)a;
    const Item *ib = (const Item*)b;
    return (ib->w - ia->w);
}

/* Koristen je pseudoslucajni generator xorshift64* koji generise 32 bitni nasumican broj */
static unsigned long long rng_state = 88172645463325252ULL; 

static unsigned int rng_u32(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (unsigned int)((rng_state * 2685821657736338717ULL) >> 32);
}
static int rand_int(int lo, int hi) { //generise se cijeli broj u intervalu od lo do hi
    unsigned int r = rng_u32();
    return lo + (int)(r % (unsigned int)(hi - lo + 1));
}

static long long sum_weights(const Item *a, int n) {
    long long s = 0;
    for (int i = 0; i <n; i++) s += a[i].w;
    return s;
}

static long long sum_selected(const Item *a, int n, const int *sel) {
    long long s = 0;
    for (int i = 0; i<n; i++) if (sel[i]) s += a[i].w;
    return s;
}

static int count_selected(int n, const int *sel) {
    int k = 0;
    for (int i = 0; i < n; i++) if (sel[i]) k++;
    return k;
}

/* Generator instanci (samo tezine) prateci Kellerer knjigu */
static void gen_weights(Item *a, int n, const char *family, int R) {
    for (int i = 0; i < n; i++) {
        int w;
        if (strcmp(family, "uncorr") == 0 || strcmp(family, "subset") == 0) {
            w = rand_int(1, R);
        } else if (strcmp(family, "simw") == 0) {
            w = rand_int(100000, 100100);
        } else {
            fprintf(stderr, "Unknown family '%s'. Use uncorr|subset|simw.\n", family);
            exit(1);
        }
        a[i].w = w;
        a[i].idx = i;
    }
}

/* Pohlepni algoritam koji bira najvecu tezinu */
static int greedy_largest_first(const Item *sorted_desc, int n, long long W, int *sel_out /* obim dimenzije n*/) {
    memset(sel_out, 0, sizeof(int) * n);
    long long s = 0;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (s >= W) break;
        sel_out[sorted_desc[i].idx] = 1;
        s += sorted_desc[i].w;
        k++;
    }
    if (s < W) return INF; // Ako nije moguce pokriti W vrati veliku vrijednost
    return k;
}

/* ILP Branch and Bound*/
typedef struct {
    const Item *it;      // predmeti sortirani po tezini
    int n;               // broj predmeta
    long long W;         // kapacitet
    int bestK;           // trenutno najbolje rjesenje
    int *bestSel;        // rjesenje koje je najbolje i trenutno izabrano
    int *curSel;         // rjesenje koje je trenutno izabrano
    long long *suffixSum; // suffixSum[i] = sum_{j=i..n-1} it[j].w
} BnB;
/* Donja granica: minimalan broj dodatnih predmeta */
static int lower_bound_additional_items(const Item *it, int n, int start, long long remW) {
    if (remW <= 0) return 0;
    long long s = 0;
    int cnt = 0;
    for (int i = start; i < n; i++) {
        s += it[i].w;
        cnt++;
        if (s >= remW) return cnt;
    }
    return INF;
}
/* Rekurzivna BnB pretraga */
static void bnb_dfs(BnB *b, int i, long long curW, int curK) {
    /* Ako je trenutna suma tezina vec >= W, imamo izvodljivo rjesenje */
    if (curW >= b->W) {
        /* Sacuvamo ovo rjesenje ako koristi manje predmeta nego do sada najbolje */
        if (curK < b->bestK) {
            b->bestK = curK;
            memcpy(b->bestSel, b->curSel, sizeof(int) * b->n);
        }
        return;
    }
    if (i >= b->n) return; // Ako smo dosli do kraja liste a nismo pokrili W onda je neuspjesna grana
    if (curK >= b->bestK) return; // Ako je broj izabranih predmeta imamo isto ili vise predmeta nego u najboljem rjesenju, nema smisla nastavljati 

    if (curW + b->suffixSum[i] < b->W) return; //Cak i ako uzmemo sve preostale predmete, ne možemo dostici W.

    long long remW = b->W - curW;
    int lb = lower_bound_additional_items(b->it, b->n, i, remW);
    if (lb == INF) return;
    if (curK + lb >= b->bestK) return;

    // Grana 1: uzmi predmet i
    b->curSel[b->it[i].idx] = 1;
    bnb_dfs(b, i + 1, curW + b->it[i].w, curK + 1);
    b->curSel[b->it[i].idx] = 0;

    // Grana 2: preskoci predmet i
    bnb_dfs(b, i + 1, curW, curK);
}

static int exact_bnb_opt(const Item *sorted_desc, int n, long long W, int *sel_out) {
    BnB b;
    b.it = sorted_desc;
    b.n = n;
    b.W = W;
    b.bestK = INF;

    b.bestSel = (int*)calloc(n, sizeof(int));
    b.curSel  = (int*)calloc(n, sizeof(int));
    b.suffixSum = (long long*)calloc(n + 1, sizeof(long long));

    // Sufiks sume (gornja granica dostupne tezine)
    b.suffixSum[n] = 0;
    for (int i = n - 1; i >= 0; i--) {
        b.suffixSum[i] = b.suffixSum[i + 1] + sorted_desc[i].w;
    }

    // Pocetnja gornja granica od pohlepnog algoritma, sto pomaze brzem odsijecanju grana
    int *tmpSel = (int*)calloc(n, sizeof(int));
    int gK = greedy_largest_first(sorted_desc, n, W, tmpSel);
    if (gK < b.bestK) {
        b.bestK = gK;
        memcpy(b.bestSel, tmpSel, sizeof(int) * n);
    }
    free(tmpSel);

    bnb_dfs(&b, 0, 0, 0);

    memcpy(sel_out, b.bestSel, sizeof(int) * n);

    int best = b.bestK;
    free(b.bestSel);
    free(b.curSel);
    free(b.suffixSum);
    return best;
}

/* Parsiranje csv fajla */
static int is_csv_filename(const char *s) {
    if (!s) return 0;
    size_t L = strlen(s);
    return (L >= 4 && strcmp(s + (L - 4), ".csv") == 0);
}

/* Struktura po familiji i alfa vrijednosti za statistiku */
typedef struct {
    double sumGreedyMs;
    double sumOptMs;
    double sumGapPct;
    int optimalGreedyCount;
    int totalRuns;
} Stats;

static void stats_init(Stats *st) {
    memset(st, 0, sizeof(*st));
}

static void stats_add(Stats *st, double gMs, double oMs, int kg, int ko) {
    st->sumGreedyMs += gMs;
    st->sumOptMs += oMs;
    if (ko > 0 && kg < INF && ko < INF) {
        st->sumGapPct += 100.0 * ((double)(kg - ko) / (double)ko);
        if (kg == ko) st->optimalGreedyCount += 1;
    }
    st->totalRuns += 1;
}

static void print_header_table(int alphaCount, const double *alphas) {
    printf("-------------------------------------------------------------------------------------------------------------\n");
    printf("| Inst | Family  |   sum(w)   |");
    for (int a = 0; a < alphaCount; a++) {
        printf("  alpha=%.2f  |", alphas[a]);
    }
    printf("\n");
    printf("|      |         |            |");
    for (int a = 0; a < alphaCount; a++) {
        printf(" kg/ko  gap%%  tg(ms)/to(ms) |");
    }
    printf("\n");
    printf("-------------------------------------------------------------------------------------------------------------\n");
}

static void print_row(int inst, const char *family, long long sumw,
                      int alphaCount, const double *alphas,
                      const int *kg, const int *ko,
                      const double *gap, const double *tg, const double *to) {
    printf("| %4d | %-7s | %9lld |", inst, family, sumw);
    for (int a = 0; a < alphaCount; a++) {
        printf(" %2d/%-2d %6.2f %7.2f/%7.2f |", kg[a], ko[a], gap[a], tg[a], to[a]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int n = 60;
    int R = 1000;
    int trials = 20;
    unsigned long long seed = (unsigned long long)time(NULL);
    const char *familyArg = "all";

    // lista alfa vrijednosti
    double alphas[32];
    int alphaCount = 0;

    const char *csvFile = NULL;

    // Fiksni pocetni argumenti
    if (argc >= 2) 
        n = atoi(argv[1]);
    if (argc >= 3) 
        R = atoi(argv[2]);
    if (argc >= 4) 
        trials = atoi(argv[3]);
    if (argc >= 5) 
        seed = strtoull(argv[4], NULL, 10);
    if (argc >= 6) 
        familyArg = argv[5];

    // Preostali argumenti, tacnije alfa vrijendosti i (ili bez) csv fajl
    for (int i = 6; i < argc; i++) {
        if (is_csv_filename(argv[i])) {
            csvFile = argv[i];
        } else {
            if (alphaCount < 32) {
                alphas[alphaCount++] = atof(argv[i]);
            }
        }
    }
    if (alphaCount == 0) {
        alphas[alphaCount++] = 0.5; // podrazumijevano (default)
    }

    rng_state = seed ? seed : 1ULL;

    // Izvrsavanje familija
    const char *families[3] = {"uncorr", "subset", "simw"};
    int famCount = 0;
    const char *runFamilies[3];

    if (strcmp(familyArg, "all") == 0) {
        runFamilies[0] = "uncorr";
        runFamilies[1] = "subset";
        runFamilies[2] = "simw";
        famCount = 3;
    } else {
        runFamilies[0] = familyArg;
        famCount = 1;
    }

    printf("Cover Knapsack Evaluation (Greedy vs Optimal ILP(BnB))\n");
    printf("n=%d, R=%d, trials=%d, seed=%llu, families=%s\n", n, R, trials, seed, familyArg);
    printf("alphas: ");
    for (int a = 0; a < alphaCount; a++) printf("%.2f%s", alphas[a], (a+1<alphaCount)?", ":"");
    printf("\n\n");

    FILE *csv = NULL;
    if (csvFile) {
        csv = fopen(csvFile, "w");
        if (!csv) {
            fprintf(stderr, "ERROR: cannot open CSV file '%s'\n", csvFile);
            return 1;
        }
        fprintf(csv, "instance,family,sumw,alpha,W,kg,ko,gap_pct,tg_ms,to_ms\n");
    }

    // Statistika za svaku familiju po alfa
    Stats stats[3][32];
    for (int f = 0; f < 3; f++)
        for (int a = 0; a < 32; a++)
            stats_init(&stats[f][a]);

    for (int f = 0; f < famCount; f++) {
        const char *fam = runFamilies[f];

        printf("=== FAMILY: %s ===\n", fam);
        print_header_table(alphaCount, alphas);

        for (int t = 1; t <= trials; t++) {
            Item *a = (Item*)calloc(n, sizeof(Item));
            gen_weights(a, n, fam, R);
            long long sumw = sum_weights(a, n);

            // Kreiraj i sortiraj kopiju predmeta po opadajucoj tezini (za greedy i BnB)
            Item *sorted = (Item*)calloc(n, sizeof(Item));
            memcpy(sorted, a, sizeof(Item) * n);
            qsort(sorted, n, sizeof(Item), cmp_desc_w);

            // Nizovi rezultata po alfa vrijednostima (greedy, optimalno, gap i vremena)
            int *kg = (int*)calloc(alphaCount, sizeof(int));
            int *ko = (int*)calloc(alphaCount, sizeof(int));
            double *gap = (double*)calloc(alphaCount, sizeof(double));
            double *tg = (double*)calloc(alphaCount, sizeof(double));
            double *to = (double*)calloc(alphaCount, sizeof(double));

            for (int ai = 0; ai < alphaCount; ai++) {
                double alpha = alphas[ai];
                long long W = (long long)(alpha * (double)sumw);
                if (W < 1) W = 1;
                if (W > sumw) W = sumw;

                int *selG = (int*)calloc(n, sizeof(int));
                int *selO = (int*)calloc(n, sizeof(int));

                clock_t g0 = clock();
                kg[ai] = greedy_largest_first(sorted, n, W, selG);
                clock_t g1 = clock();

                clock_t o0 = clock();
                ko[ai] = exact_bnb_opt(sorted, n, W, selO);
                clock_t o1 = clock();

                tg[ai] = 1000.0 * (double)(g1 - g0) / (double)CLOCKS_PER_SEC;
                to[ai] = 1000.0 * (double)(o1 - o0) / (double)CLOCKS_PER_SEC;

                if (ko[ai] > 0 && kg[ai] < 1000000000 && ko[ai] < 1000000000) {
                    gap[ai] = 100.0 * ((double)(kg[ai] - ko[ai]) / (double)ko[ai]);
                } else {
                    gap[ai] = 0.0;
                }
                // Mapiraj indekse statistike za familiju
                int famIdx = 0;
                if (strcmp(fam, "uncorr") == 0) famIdx = 0;
                else if (strcmp(fam, "subset") == 0) famIdx = 1;
                else if (strcmp(fam, "simw") == 0) famIdx = 2;

                stats_add(&stats[famIdx][ai], tg[ai], to[ai], kg[ai], ko[ai]);

                if (csv) {
                    fprintf(csv, "%d,%s,%lld,%.6f,%lld,%d,%d,%.6f,%.6f,%.6f\n",
                            t, fam, sumw, alpha, W, kg[ai], ko[ai], gap[ai], tg[ai], to[ai]);
                }

                free(selG);
                free(selO);
            }

            print_row(t, fam, sumw, alphaCount, alphas, kg, ko, gap, tg, to);

            free(kg);
            free(ko);
            free(gap);
            free(tg);
            free(to);
            free(sorted);
            free(a);
        }

        printf("-------------------------------------------------------------------------------------------------------------\n");
        printf("\n");
    }
    printf("\n==================== SUMMARY ====================\n");
    for (int f = 0; f < famCount; f++) {
        const char *fam = runFamilies[f];
        int famIdx = 0;
        if (strcmp(fam, "uncorr") == 0) famIdx = 0;
        else if (strcmp(fam, "subset") == 0) famIdx = 1;
        else if (strcmp(fam, "simw") == 0) famIdx = 2;

        printf("Family: %s\n", fam);
        for (int ai = 0; ai < alphaCount; ai++) {
            Stats *st = &stats[famIdx][ai];
            int runs = st->totalRuns;
            double avgG = st->sumGreedyMs / (double)runs;
            double avgO = st->sumOptMs / (double)runs;
            double avgGap = st->sumGapPct / (double)runs;
            double optPct = 100.0 * (double)st->optimalGreedyCount / (double)runs;

            printf("  alpha=%.2f | avg tg=%.4f ms | avg to=%.4f ms | avg gap=%.3f%% | greedy optimal=%.2f%%\n",
                   alphas[ai], avgG, avgO, avgGap, optPct);
        }
        printf("\n");
    }

    if (csv) {
        fclose(csv);
        printf("CSV exported to: %s\n", csvFile);
    }

    return 0;
}