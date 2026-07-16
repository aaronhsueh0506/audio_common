/*
 * atomic_symlink_swap -- atomic publish-time filesystem helpers.
 *
 * Mode 1 (round-5 review P2: publish's `current` handoff swap):
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
 * Mode 2 (round-6 review P2: one-event-one-file ATTEST install):
 *
 *   swapln --excl-install <src-file> <dst-path>
 *
 * Copies <src-file>'s full content to <dst-path>.<pid>.tmp, then link(2)s the
 * temp onto <dst-path>. link(2) fails with EEXIST if <dst-path> already
 * exists (atomic no-clobber -- the kernel-level equivalent of
 * open(O_CREAT|O_EXCL), but installing complete content in one step so the
 * final name is only ever observable with its full bytes, never partially
 * written). Exit status: 0 on success, 2 if <dst-path> already exists (the
 * caller regenerates its event id and retries -- it must NEVER overwrite),
 * 1 on any other failure. The temp is unlinked on every path.
 *
 * No fsync: consistent with every other install in this build system
 * (MANIFEST, config.manifest, archives -- all temp+atomic-rename without
 * fsync). Durability across power loss was never part of the contract;
 * atomicity against concurrent readers/writers is.
 *
 * Both modes embed the PID in the temp name (no fixed-name collision between
 * concurrent callers). Compiled at publish time with the HOST compiler
 * (HOSTCC, default `cc`) -- never $(CC), which may be a cross compiler whose
 * output cannot run on the build host. See each repo's Makefile `publish`
 * recipe.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/* Builds "<path>.<pid>.tmp" into buf and clears any stale same-PID temp (it
 * can only be our own, from a previous failed run of this exact path in this
 * exact process id -- safe to clear). Shared by both modes so the temp-naming
 * scheme cannot drift between them. */
static int make_tmp_path(char* buf, size_t bufsz, const char* path) {
    int n = snprintf(buf, bufsz, "%s.%ld.tmp", path, (long)getpid());
    if (n < 0 || (size_t)n >= bufsz) {
        fprintf(stderr, "swapln: path too long\n");
        return -1;
    }
    (void)unlink(buf);
    return 0;
}

static int swap_symlink(const char* target, const char* linkpath) {
    char tmp[4096];
    if (make_tmp_path(tmp, sizeof(tmp), linkpath) != 0)
        return 1;

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

static int excl_install(const char* src, const char* dst) {
    char tmp[4096];
    if (make_tmp_path(tmp, sizeof(tmp), dst) != 0)
        return 1;

    int in = open(src, O_RDONLY);
    if (in < 0) {
        fprintf(stderr, "swapln: open(%s): %s\n", src, strerror(errno));
        return 1;
    }

    int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        fprintf(stderr, "swapln: open(%s): %s\n", tmp, strerror(errno));
        close(in);
        return 1;
    }

    char buf[65536];
    ssize_t r;
    while ((r = read(in, buf, sizeof(buf))) > 0) {
        ssize_t off = 0;
        while (off < r) {
            ssize_t w = write(out, buf + off, (size_t)(r - off));
            if (w < 0) {
                fprintf(stderr, "swapln: write(%s): %s\n", tmp, strerror(errno));
                close(in); close(out); (void)unlink(tmp);
                return 1;
            }
            off += w;
        }
    }
    close(in);
    if (r < 0 || close(out) != 0) {
        fprintf(stderr, "swapln: copying %s failed: %s\n", src, strerror(errno));
        (void)unlink(tmp);
        return 1;
    }

    if (link(tmp, dst) != 0) {
        int rc = (errno == EEXIST) ? 2 : 1;
        if (rc != 2)
            fprintf(stderr, "swapln: link(%s, %s): %s\n", tmp, dst, strerror(errno));
        (void)unlink(tmp);
        return rc;
    }
    (void)unlink(tmp);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && strcmp(argv[1], "--excl-install") == 0)
        return excl_install(argv[2], argv[3]);
    if (argc == 3 && argv[1][0] != '-')
        return swap_symlink(argv[1], argv[2]);
    fprintf(stderr,
            "usage: swapln <link-target> <link-path>\n"
            "       swapln --excl-install <src-file> <dst-path>\n");
    return 1;
}
