
#include <sys/time.h>

#define SEC 100000L

#define start_stopper(value){\
	value.it_value.tv_sec = SEC; \
	value.it_value.tv_usec= 0; \
	value.it_interval.tv_sec = SEC; \
	value.it_interval.tv_usec= 0; \
	setitimer(ITIMER_PROF, &value, NULL);}

#define elapsed_stopper_time(time, value, ovalue){\
	getitimer(ITIMER_PROF, &value); \
	time = SEC - (double)(value.it_value.tv_sec) \
           - (double)(value.it_value.tv_usec)/1000000; \
	setitimer(ITIMER_PROF, &value, &ovalue); \
  }

