#include <time.h>
#include <stdio.h>
typedef struct {
    clock_t start;
    clock_t end;
} Timer;

void startTimer(Timer* t) {
    t->start = clock();
}

void stopTimer(Timer* t) {
    t->end = clock();
}

void printElapsedTime(Timer t, const char* msg) {
    double time = (double)(t.end - t.start) / CLOCKS_PER_SEC;
    printf("%s %f sec\n", msg, time);
}