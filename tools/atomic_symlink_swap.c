/*
 * atomic_symlink_swap -- truly atomic cross-platform symlink replacement
 * (round-5 review P2: publish's `current` handoff swap).
 *
 *   swapln <link-target> <link-path>
 *
 * Creates the symlink <link-path>.<pid>.tmp -> <link-target>, then rename(2)s
 * it onto <link-path>. rename(2) NEVER follows symbolic links in either
 * argument (POSIX; verified semantics on both macOS and GNU/Linux), so this
 * atomically replaces an existing <link-path> symlink in one syscall -- no
 * observable window where <link-path> is missing, and no possibility of the
 * temp link being moved INTO the directory the old symlink points at (the
 * userland `mv` failure mode this helper replaces: mv stat()s -- i.e.
 * FOLLOWS -- a symlink-to-directory destination and appends the source's
 * basename to it; GNU mv only avoids that with -T, which BSD/macOS mv does
 * not have, and the previous rm+mv fallback left a window with no `current`
 * at all and went destructive on ANY mv failure).
 *
 * The temp name embeds the PID (no fixed-name collision between concurrent
 * callers) and is unlinked on every failure path, so no .tmp residue can
 * survive this process. Exit status: 0 on success, 1 on any failure (with a
 * one-line diagnostic on stderr).
 *
 * Compiled at publish time with the HOST compiler (HOSTCC, default `cc`) --
 * never $(CC), which may be a cross compiler whose output cannot run on the
 * build host. See each repo's Makefile `publish` recipe.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: swapln <link-target> <link-path>\n");
        return 1;
    }
    const char* target = argv[1];
    const char* linkpath = argv[2];

    char tmp[4096];
    int n = snprintf(tmp, sizeof(tmp), "%s.%ld.tmp", linkpath, (long)getpid());
    if (n < 0 || (size_t)n >= sizeof(tmp)) {
        fprintf(stderr, "swapln: link path too long\n");
        return 1;
    }

    /* A stale same-PID temp can only be our own from a previous failed run
     * of this exact path in this exact process id -- safe to clear. */
    (void)unlink(tmp);

    if (symlink(target, tmp) != 0) {
        fprintf(stderr, "swapln: symlink(%s, %s): %s\n", target, tmp, strerror(errno));
        return 1;
    }
    if (rename(tmp, linkpath) != 0) {
        fprintf(stderr, "swapln: rename(%s, %s): %s\n", tmp, linkpath, strerror(errno));
        (void)unlink(tmp);
        return 1;
    }
    return 0;
}
