#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
int main () {
  pid_t child_pid;

  
  printf("Main process id = %d (parent PID = %d)\n \n",
(int) getpid(), (int)  getppid());

int add_ten = (int) getpid() + 10;

  child_pid = fork();

  

  if (child_pid != 0){
    printf("This is the parent process. \n");
    printf("Parent: child's process id = %d\n ", child_pid);
  }
  else{
    printf("This is the child process. \n");
    printf("Child:  my process id = %d\n ", (int) getpid());

    printf("Child_pid = %d\n ", child_pid);
  }

  printf("Hello world! by process_id : %d add_ten is: %d.\n \n", getpid(), add_ten);
  return 0;
}