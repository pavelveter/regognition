// Package cache — SQLite-backed cache for face embeddings.
//
// A FaceCache wraps the (path → embeddings) mapping produced by an
// Embedder across many runs of the regognition pipeline over the same
// photo archive.
//
// The store is keyed by path; the SHA1 of (path|mtime|size) is stored
// alongside the result so a file rewrite triggers cache invalidation
// without operator intervention.
//
// Schema matches TODO.md §3 verbatim; only an opportunistic index on
// hash is added to make decisions about a possible "rebuild" step
// cheap.
package cache

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	_ "modernc.org/sqlite"
)

// DriverName is the database/sql driver name registered by modernc/sqlite.
const DriverName = "sqlite"

// FaceCache is a stable (path, fileHash) -> []face-vectors store.
//
// Implementations: SQLite (default). Tightly coupled to the embedding
// dimensionality via `EmbedDim`; see codec.go.
type FaceCache interface {
	// Get returns the stored hash + faces for key, or (ok=false) on miss.
	// Callers MUST compare the returned hash against a freshly-computed
	// file hash and treat a mismatch as a miss (the file has changed).
	Get(ctx context.Context, key string) (hash string, faces [][]float32, ok bool, err error)
	// Set writes/overwrites the (key, hash, faces) tuple.
	Set(ctx context.Context, key string, hash string, faces [][]float32) error
	Close() error
}

// sqliteCache is the modernc.org/sqlite-backed FaceCache.
type sqliteCache struct {
	db     *sql.DB
	path   string
	mu     sync.Mutex
	closed bool
}

// Open creates or opens a SQLite database at dbPath, applies the
// project schema, and enables WAL for inter-process / inter-thread
// read concurrency while writes are serialized.
func Open(dbPath string) (FaceCache, error) {
	if dbPath == "" {
		return nil, errors.New("cache: empty database path")
	}
	abs, err := filepath.Abs(dbPath)
	if err != nil {
		return nil, fmt.Errorf("cache: resolve path: %w", err)
	}
	db, err := sql.Open(DriverName, abs)
	if err != nil {
		return nil, fmt.Errorf("cache: open sqlite: %w", err)
	}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("cache: ping: %w", err)
	}
	// WAL mode allows concurrent readers while writes are serialized.
	// SetMaxOpenConns(0) means unlimited connections from the pool,
	// which is safe with WAL + mutex on writes.
	db.SetMaxOpenConns(0)

	if _, err := db.Exec(`PRAGMA journal_mode = WAL`); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("cache: WAL: %w", err)
	}
	if _, err := db.Exec(`PRAGMA synchronous = NORMAL`); err != nil {
		// Non-fatal: continue with the driver default.
	}

	schema := `
CREATE TABLE IF NOT EXISTS photo_cache (
    path          TEXT PRIMARY KEY,
    hash          TEXT NOT NULL,
    faces_count   INTEGER NOT NULL,
    embeddings    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photo_cache_hash ON photo_cache(hash);
`
	if _, err := db.Exec(schema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("cache: schema: %w", err)
	}
	return &sqliteCache{db: db, path: abs}, nil
}

// Get implements the FaceCache interface.
func (s *sqliteCache) Get(ctx context.Context, key string) (string, [][]float32, bool, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT hash, faces_count, embeddings FROM photo_cache WHERE path = ? LIMIT 1`,
		key,
	)
	var storedHash string
	var count int64
	var blob []byte
	if err := row.Scan(&storedHash, &count, &blob); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil, false, nil
		}
		return "", nil, false, fmt.Errorf("cache: get: %w", err)
	}
	faces, err := decodeEmbeddings(blob, int(count))
	if err != nil {
		return storedHash, nil, false, err
	}
	return storedHash, faces, true, nil
}

// Set implements the FaceCache interface.
func (s *sqliteCache) Set(ctx context.Context, key, hash string, faces [][]float32) error {
	blob := encodeEmbeddings(faces)
	_, err := s.db.ExecContext(ctx,
		`INSERT OR REPLACE INTO photo_cache (path, hash, faces_count, embeddings) VALUES (?, ?, ?, ?)`,
		key, hash, len(faces), blob,
	)
	if err != nil {
		return fmt.Errorf("cache: set: %w", err)
	}
	return nil
}

// Close implements the FaceCache interface. Idempotent.
func (s *sqliteCache) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.db.Close()
}
