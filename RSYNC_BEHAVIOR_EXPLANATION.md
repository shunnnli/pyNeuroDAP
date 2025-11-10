# Rsync Behavior with Wildcards - Explanation

## Your Questions Answered

### Q1: Will `*` match all folders starting with "findr-results-"?
**Answer: YES**, but with important caveats:

- The wildcard `findr-results-*` will match **ALL** directories/files starting with "findr-results-" on the remote server
- When using rsync over SSH, the wildcard expansion happens on the **remote server**
- If multiple directories match, rsync will try to copy **all of them**

### Q2: If local directory has extra files, will they be overwritten?
**Answer: NO, extra files are preserved, but existing files are updated:**

With `rsync -av` (archive mode):
- ✅ **Extra files in destination are NOT deleted** (unless you use `--delete`)
- ✅ **Files that exist in both locations are updated** if source is newer (by default)
- ✅ **Directories are merged** - existing files are updated, not the entire directory overwritten
- ✅ **Permissions and timestamps are preserved**

## The Problem with Your Original Command

Your original command had a mismatch:
```bash
# Source: matches ALL directories (findr-results-*)
# Destination: specific directory (findr-results-20251109/)
rsync -av 'server/findr-results-*' 'local/findr-results-20251109/'
```

**Issues:**
1. If multiple `findr-results-*` directories exist on server, all will be copied into the destination
2. The destination path suggests you only want ONE specific directory
3. This can cause confusion and unexpected behavior

## The Fix

I've updated the notebook to provide two options:

### Option 1: Sync Specific Directory (Recommended)
```bash
rsync -av 'server/findr-results-20251109/' 'local/findr-results-20251109/'
```
- ✅ Syncs only the specific directory matching your `analysispath`
- ✅ Clear and predictable behavior
- ✅ Matches the directory you created on the server

### Option 2: Sync ALL Results Directories
```bash
rsync -av 'server/findr-results-*/' 'local/parent/directory/'
```
- ✅ Syncs all `findr-results-*` directories
- ✅ Copies them into the parent directory (not a specific subdirectory)
- ⚠️ Use only if you want to sync multiple result directories at once

## Rsync Behavior Details

### Default Behavior (`rsync -av`)
- **Merges directories** - doesn't delete existing files
- **Updates files** - overwrites if source is newer or different
- **Preserves attributes** - permissions, timestamps, etc.

### If You Want Different Behavior

**Delete files in destination that don't exist in source:**
```bash
rsync -av --delete 'source/' 'destination/'
```
⚠️ **Warning:** This will delete extra files in destination!

**Only update if source is newer:**
```bash
rsync -avu 'source/' 'destination/'  # -u = update (skip newer files in dest)
```

**Dry run (see what would happen without actually syncing):**
```bash
rsync -avn 'source/' 'destination/'  # -n = dry run
```

**Show progress:**
```bash
rsync -av --progress 'source/' 'destination/'
```

## Best Practices

1. **Use specific paths** instead of wildcards when possible
2. **Test with `--dry-run`** (`-n` flag) before syncing
3. **Use trailing slashes carefully:**
   - `source/` → copies contents of source into destination
   - `source` → copies source directory into destination
4. **Match your analysis path** - if you create `findr-results-20251109` on server, sync that specific directory

## Example: Safe Sync Workflow

```bash
# 1. Dry run to see what would happen
rsync -avn --progress 'server/findr-results-20251109/' 'local/findr-results-20251109/'

# 2. If it looks good, do the actual sync
rsync -av --progress 'server/findr-results-20251109/' 'local/findr-results-20251109/'

# 3. Verify the sync
ls -la local/findr-results-20251109/
```

## Summary

- ✅ Wildcard `*` matches all directories starting with the pattern
- ✅ Extra files in local directory are **NOT deleted** (unless `--delete` is used)
- ✅ Existing files are **updated** if source is newer/different
- ✅ Use **specific directory paths** instead of wildcards for clarity and safety
- ✅ The updated notebook now uses the specific directory path matching your `analysispath`

