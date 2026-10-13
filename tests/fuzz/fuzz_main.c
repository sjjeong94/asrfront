/* Driver for fuzz_load.c without libFuzzer (e.g. GCC / MSVC builds):
 *
 *   fuzz_replay FILE_OR_DIR...                  run every input once (corpus regression test)
 *   fuzz_replay -mutate N [-seed S] FILE_OR_DIR...  also run N random mutations of each input
 *
 * The mutations are a cheap stand-in for libFuzzer's (byte / bit / 0x00 / 0xFF edits, often in
 * the header, and truncations); use the libFuzzer build for real fuzzing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size);

static unsigned long long rng_state = 88172645463325252ULL;
static long mutations = 0;
static size_t n_inputs = 0;

static unsigned rnd(void) { /* xorshift64 */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (unsigned)(rng_state >> 16);
}

static void run_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    unsigned char *buf = NULL;
    size_t size = 0, cap = 0, got;
    do {
        if (size == cap) {
            cap = cap ? 2 * cap : 4096;
            unsigned char *nb = realloc(buf, cap);
            if (!nb)
                exit(2);
            buf = nb;
        }
        got = fread(buf + size, 1, cap - size, f);
        size += got;
    } while (got > 0);
    fclose(f);
    LLVMFuzzerTestOneInput(buf, size);
    n_inputs++;

    unsigned char *m = malloc(size > 0 ? size : 1);
    for (long it = 0; it < mutations && size > 0; it++) {
        memcpy(m, buf, size);
        size_t len = size;
        int edits = 1 + (int)(rnd() % 8);
        for (int e = 0; e < edits; e++) {
            unsigned r = rnd() % 10;
            size_t span = r < 7 && size > 256 ? 256 : size; /* headers are where parsers live */
            size_t pos = rnd() % span;
            if (r < 4)
                m[pos] = (unsigned char)rnd();
            else if (r < 6)
                m[pos] = (r & 1) ? 0xFF : 0x00;
            else if (r < 8)
                m[pos] ^= (unsigned char)(1u << (rnd() % 8));
            else
                len = pos + 1 < len ? pos + 1 : len;
        }
        LLVMFuzzerTestOneInput(m, len);
        n_inputs++;
    }
    free(m);
    free(buf);
}

static void run_path(const char *path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) {
        char pattern[MAX_PATH];
        snprintf(pattern, sizeof pattern, "%s\\*", path);
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(pattern, &fd);
        if (h == INVALID_HANDLE_VALUE)
            return;
        do {
            if (fd.cFileName[0] == '.')
                continue;
            char child[MAX_PATH];
            snprintf(child, sizeof child, "%s\\%s", path, fd.cFileName);
            run_path(child);
        } while (FindNextFileA(h, &fd));
        FindClose(h);
        return;
    }
#else
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        struct dirent *e;
        while (d && (e = readdir(d)) != NULL) {
            if (e->d_name[0] == '.')
                continue;
            size_t len = strlen(path) + strlen(e->d_name) + 2;
            char *child = malloc(len);
            if (!child)
                exit(2);
            snprintf(child, len, "%s/%s", path, e->d_name);
            run_path(child);
            free(child);
        }
        if (d)
            closedir(d);
        return;
    }
#endif
    run_file(path);
}

int main(int argc, char **argv) {
    int i = 1;
    for (; i + 1 < argc && argv[i][0] == '-'; i += 2) {
        if (strcmp(argv[i], "-mutate") == 0)
            mutations = atol(argv[i + 1]);
        else if (strcmp(argv[i], "-seed") == 0)
            rng_state ^= strtoull(argv[i + 1], NULL, 10) * 0x9E3779B97F4A7C15ULL;
        else
            break;
    }
    if (i >= argc) {
        fprintf(stderr, "usage: %s [-mutate N] [-seed S] FILE_OR_DIR...\n", argv[0]);
        return 2;
    }
    for (; i < argc; i++)
        run_path(argv[i]);
    printf("fuzz_replay: %zu inputs ok\n", n_inputs);
    return 0;
}
