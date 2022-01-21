/**
 * @file Timer.cpp
 * 
 * @brief CSS432 UDP Socket Class
 *
 * @ingroup Simulator/Utils
 */

#include "Timer.h"

// Constructor ----------------------------------------------------------------
Timer::Timer() {
	startTime.tv_sec = 0;
	startTime.tv_usec = 0;
	endTime.tv_sec = 0;
	endTime.tv_usec = 0;
}

// Memorize the current time in startTime -------------------------------------
void Timer::start() { gettimeofday(&startTime, nullptr); }

// Get the diff between the start and the current time (in microseconds)
long Timer::lap() {
	gettimeofday(&endTime, nullptr);
	long interval =
		(endTime.tv_sec - startTime.tv_sec) * 1000000 +
		(endTime.tv_usec - startTime.tv_usec);
	return interval;
}

// Get the diff between the old and the specified time (in microseconds)
long Timer::lap(long oldTv_sec, long oldTv_usec) {
	gettimeofday(&endTime, nullptr);
	long interval =
		(endTime.tv_sec - oldTv_sec) * 1000000 +
		(endTime.tv_usec - oldTv_usec);
	return interval;
}

// Get sec --------------------------------------------------------------------
long Timer::getSec() { return startTime.tv_sec; }

// Get usec -------------------------------------------------------------------
long Timer::getUsec() { return startTime.tv_usec; }
