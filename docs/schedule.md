# Schedule

VM Scheduler schedules timings for Proc to execute.

## Queues

VM Scheduler maintains two queues:

* `ready_queue`.
* `waiting_queue`.

The rules are described below:

* Scheduler picks the HEAD of `ready_queue` to run inside VM for a given time slice.
* Scheduler moves the head of `ready_queue` to the END of `ready_queue` if the Proc can't finish in a given time slice.
* Scheduler moves the HEAD of `ready_queue` to the END of `waiting_queue` if the Proc is blocked by `recv` or I/O.
* Scheduler moves a Proc from `waiting_queue` to `ready_queue` if it receives a message or waiting timeout.

Note that at a specific moment, there is only one instruction of a Proc is actually running in single CPU core. On a multiple CPU core machine, VM can create multiple scheduler and hence multiple Proc 

## Schedule Loop

Schedule loop keeps running forever.

The rules for each loop round are described below:

* Migrate Proc to other VM if needed.
* 
* Pick a Proc to execute.
* The Proc can execute `maximum_instructions`.
* If Proc can finish, free Proc and go to next loop
* If Proc can't finish, suspend Prod, move it to the END of `ready_queue`.
