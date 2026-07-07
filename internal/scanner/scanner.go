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

// WalkOptions configures the directory walker with filters.
type WalkOptions struct {
	// SkipDirs are directory base names to skip (exact match, case-insensitive).
	SkipDirs []string
	// SkipDirPatterns are glob patterns for directory names to skip
	// (e.g. ".*", "_*"). Matched via filepath.Match against d.Name().
	SkipDirPatterns []string
	// SkipFiles are glob patterns for file names to skip
	// (e.g. "._*", "*.tmp"). Matched via filepath.Match against basename.
	SkipFiles []string
	// Extensions are allowed file extensions (e.g. ".jpg", ".jpeg").
	// Empty means use SupportedExts. Matching is case-insensitive.
	Extensions []string
}

// Walk returns absolute paths of all supported image files under root.
// The slice is sorted by absolute path. Errors inside the tree are
// returned but do not abort the walk.
func Walk(root string) ([]string, error) {
	return WalkWithSkip(root, nil)
}

// WalkWithSkip is like Walk but skips directories whose base name
// matches any entry in skipDirs. skipDirs may be nil to skip nothing.
func WalkWithSkip(root string, skipDirs []string) ([]string, error) {
	return WalkWithOptions(root, WalkOptions{SkipDirs: skipDirs})
}

// WalkWithOptions is the full-featured walker. Applies all filters from opts.
func WalkWithOptions(root string, opts WalkOptions) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, errors.New("scanner: root is not a directory: " + root)
	}

	// Build skip-dir set for O(1) exact-match lookup.
	skipSet := make(map[string]struct{}, len(opts.SkipDirs))
	for _, name := range opts.SkipDirs {
		name = strings.TrimSpace(name)
		if name != "" {
			skipSet[strings.ToLower(name)] = struct{}{}
		}
	}

	// Build allowed extension set (case-insensitive).
	extSet := make(map[string]struct{})
	for _, ext := range opts.Extensions {
		ext = strings.ToLower(strings.TrimSpace(ext))
		if ext != "" {
			extSet[ext] = struct{}{}
		}
	}
	// Fall back to SupportedExts if no explicit extensions.
	if len(extSet) == 0 {
		for ext := range SupportedExts {
			extSet[ext] = struct{}{}
		}
	}

	var paths []string
	walkErr := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if d == nil {
				return nil
			}
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			nameLower := strings.ToLower(name)
			// Exact match (backward-compatible).
			if _, skip := skipSet[nameLower]; skip {
				return filepath.SkipDir
			}
			// Glob pattern match for directories.
			for _, pat := range opts.SkipDirPatterns {
				if matched, _ := filepath.Match(pat, name); matched {
					return filepath.SkipDir
				}
				// Also match against lowercase for case-insensitive glob.
				if matched, _ := filepath.Match(pat, nameLower); matched {
					return filepath.SkipDir
				}
			}
			return nil
		}

		// File skip patterns (glob match against basename).
		base := d.Name()
		baseLower := strings.ToLower(base)
		for _, pat := range opts.SkipFiles {
			if matched, _ := filepath.Match(pat, base); matched {
				return nil
			}
			if matched, _ := filepath.Match(pat, baseLower); matched {
				return nil
			}
		}

		// Extension filter (case-insensitive).
		ext := strings.ToLower(filepath.Ext(base))
		if _, ok := extSet[ext]; !ok {
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
