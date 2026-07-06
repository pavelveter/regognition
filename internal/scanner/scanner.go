// Package scanner walks a directory tree and returns paths to supported
// image files (jpg, jpeg, png, webp). Output is sorted for deterministic
// downstream behavior.
package scanner

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SupportedExts is the set of file extensions the pipeline can decode.
// Adjust if HEIC/HEIF support is added later.
var SupportedExts = map[string]struct{}{
	".jpg":  {},
	".jpeg": {},
	".png":  {},
	".webp": {},
}

// Walk returns absolute paths of all supported image files under root.
// The slice is sorted by absolute path. Errors inside the tree are
// returned but do not abort the walk.
func Walk(root string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, errors.New("scanner: root is not a directory: " + root)
	}

	var paths []string
	walkErr := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			// Don't abort for permission errors on a single subtree.
			if d == nil {
				return nil
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(d.Name()))
		if _, ok := SupportedExts[ext]; !ok {
			return nil
		}
		abs, absErr := filepath.Abs(p)
		if absErr != nil {
			return nil
		}
		paths = append(paths, abs)
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}
	sort.Strings(paths)
	return paths, nil
}
