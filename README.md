# cuckoo_hash

## configuration of machine
>>CPU: i7-12700\
>>GPU: GTX-1660s

## Algorithm
>>&emsp;&emsp;On the base of original cuckoo hash, I separate the table into funcutin_num * 2 parts and each function only take charge of two of these part. In this way, we can efficiently avoid hash conflict and insert elements fast.\
>>&emsp;&emsp;I make insert and search part into cuda version. In insert, I add a global variable need_rehash to terminate kernal when some value's evict chain has surpassed the evict_bound. In search, I terminate the kernal as long as it find the pos so that it will perform well if most to-find elements are in the range of first hash function.\
>>&emsp;&emsp;I use cpp's standard random creater so I don't need seed.

## Benchmark
>>&emsp;&emsp;Each benchmark of insert and search record the time before and after their class function is called. Insert function will return the rehash-times. So the speed is computed by (data_size * rehash_times / used_time). Because search won't cause rehash, the speed is computed by (data_size / used_time).

## Improvement and problems
>>&emsp;&emsp;There is no direct improve on insert and search function instead of the mode I use.\
>>&emsp;&emsp;I make some improvement on search data if most data are not in the hash table. In order to fast get the answer of each hash function, I launch multiple kernals at the same time to find them. It acutually speed up the situation I hope to improve, but the performance of the situation that most to-find are in the hash table. As I don't know the way to communicate between kernal functions.

## The answer to the task4
>>&emsp;&emsp;The better evict bound is when the ratio is about 5 or 6. Because it ensures enough space to insert elements and limits the infinit loop.

## Compile and run
>>&emsp;&emsp;There's a compile.sh which should be a compile file to get task1/2/3/4. And a run.sh can automatically run the experiments task.\
>>&emsp;&emsp;The basic input director is done instead of seed because of what i have said.\
>>&emsp;&emsp;Besides basic input director, I add --function_num to specify the number of hash function the table use. And --file-in to specify user-specified file, which may have some display error which is a little hard to fix.