// main.cpp
// Sharded SBert (GGML) + HNSW search with kNN / radius / relative / adaptive
// Build on macOS: link with -lbert -lggml -framework Accelerate, include hnswlib headers

#define STANDALONE 1 /* run this standalone for testing */

// NOTE: Since this code depends upon a number of libs that use modern C++ we'll loosen
// our restrictions and embrace it here.
// The core engine will still compile and run using minimal compilers but the support of
// dense vectors will just demand a modern compiler. It probably makes no sense anyway
// to want to support embeddings on these platforms as they simply won't have the memory.

// This implementation supports shards. When the configured max_elements (see HnswConfig)
// is reached (the reserved capacity) of the HNSW we start a new shard.   We have also a
// number of methods to do search in parallel on these shards. The number of shards should
// probably be at most 2 but should be under the number of CPU cores.
// We also have a method to merge the last two shards: merge_last_two()
// that works by also expanding the max_elements accordingly.
// The method merge() keeps calling merge_last_two() as long as it can, effectively
// creating a single index.
//
// For best performance, depending upon memory and cores, one should effectively limit
// the number of shards on a single machine to a reasonable number and use multiple
// machines to distribute the load.


// How to specify what fields are to be handled as dense embeddings?
// In production:
// we have two logics: inclusion and exclusion.
//    inclusion: create X type indexes for fields so defined.
//    exclusion: create X type indexes for all fields except those defined
// This is addressed by
// [DbInfo]
// DefaultFieldType=<Fieldype to use when one is not defined>
// If this is defined then any field whose fieldtype has not been defined get this as
// its fieldtype.
// So to define HNSW as the default field type it would be set as the default.
// NOTE: All fields irrespective of the types get also indexed as text.
//
// See class FIELDTYPE in attrlist.hxx as well as the code in idbobj.hxx and doctype.cxx
// 

/*
 Future improvement:

Write the name of the model used for the embeddings in the offset file.
This is important since the HNSW index and embeeding search depend upon
using the same sBert model.

format:
<magic><int8 for length><name> 
magic is a byte: see src/magic.h for list we support 
name is written without the tailing \0 if length is even. This way
the offset is always 2 aligned:  2+strlen(name) + (strlen(name) % 2)
 
NOTE: We use name and not full path as path won't be portable to another
machine.
If in the future we discover problems or a possible adverserial attack is
not just theoretical we can extend the header with a 64-bit checksum.

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <unordered_map>

#include "bert.h"
#include "hnswlib/hnswlib.h"

#ifndef MT
# define MT 1 /* compile Async search over shards methods */
#endif
#ifndef USE_FAST_BYTE_SWAP
# define USE_FAST_BYTE_SWAP 1
#endif


#if MT
# include <future>
# include <thread> /* needed only for std::thread::hardware_concurrency(); */
#endif

// These will probably change in the distribution
static const char kSentencesFileExtension[] = ".txt";
static const char kOffsetsFileExtension[]   = ".bfc";
static const char kIndexFileExtension[]     = ".hdx";
// NOTE: the sentence file is ONLY here for debuging. The offsets will later be GPs
// encoding an id to the file path and the offset addresses in the 64-bit integer..

static const float kFloatComparisonTolerance = std::numeric_limits<float>::epsilon() * 100.0f; // A common float comparison tolerance

// --------------- Portability helpers ----------

// force little-endian storage

// Instead of
//   fin.read((char*)&s,8);
// we write
//   s = read_int64(fin);
// Instead of  
//   ofs.write((char*)&s,8);
// we write    
//   write_int64(ofs, s);


// Jam these into an unamed namespace
namespace {

/*
alternative (without normed read/write)

inline int64_t read_int64(std::istream &is) {
    uint64_t u = 0;
    fin.read((char*)&s,8);
    return u;
}
inline void  write_int64(std::ostream &os, int64_t u) {
    os.write((char*)&u,8);
}

*/
#if USE_FAST_BYTE_SWAP /* Fast but less readable code */

#if defined(_MSC_VER)
    #include <intrin.h>
    #define bswap64 _byteswap_uint64
#elif defined(__clang__) || defined(__GNUC__)
    #define bswap64 __builtin_bswap64
#else
    // fallback implementation
    inline uint64_t bswap64(uint64_t x) {
        return ((x & 0x00000000000000FFULL) << 56) |
               ((x & 0x000000000000FF00ULL) << 40) |
               ((x & 0x0000000000FF0000ULL) << 24) |
               ((x & 0x00000000FF000000ULL) <<  8) |
               ((x & 0x000000FF00000000ULL) >>  8) |
               ((x & 0x0000FF0000000000ULL) >> 24) |
               ((x & 0x00FF000000000000ULL) >> 40) |
               ((x & 0xFF00000000000000ULL) >> 56);
    }
#endif

inline uint64_t to_le64(uint64_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return bswap64(x);
#else
    return x;
#endif
}

inline uint64_t from_le64(uint64_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return bswap64(x);
#else
    return x;
#endif
}

// write int64_t in little-endian
inline void write_int64(std::ostream &os, int64_t v) {
    uint64_t u = to_le64(static_cast<uint64_t>(v));
    os.write(reinterpret_cast<const char*>(&u), sizeof(u));
}

// read int64_t from little-endian
inline int64_t read_int64(std::istream &is) {
    uint64_t u = 0;
    is.read(reinterpret_cast<char*>(&u), sizeof(u));
    return static_cast<int64_t>(from_le64(u));
}


#else


inline void write_int64(std::ostream &os, int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);
    for (int i = 0; i < 8; i++) {
        char byte = static_cast<char>((u >> (i * 8)) & 0xFF); // little-endian
        os.put(byte);
    }
}

inline int64_t read_int64(std::istream &is) {
    uint64_t u = 0;
    for (int i = 0; i < 8; i++) {
        int c = is.get();
        if (c == EOF) throw std::runtime_error("Unexpected EOF while reading int64");
        u |= (static_cast<uint64_t>(c) & 0xFF) << (i * 8);
    }
    return static_cast<int64_t>(u);
}

#endif

inline void  write_int128(std::ostream &os, int64_t s, int64_t e) {
    write_int64(os, s);
    write_int64(os, e);
}


inline void write_offset_entry_for_label(std::ostream &os, size_t label, int64_t start, int64_t end) {
  os.seekp((std::streamoff)label * 2*sizeof(int64_t) /* 16 */);
  write_int128(os, start, end);
}

inline void write_empty_offset_entry (std::ostream &os) { write_int128(os, 0, 0); }

inline void write_empty_offset_entry( std::ostream &os, size_t label) {
  write_offset_entry_for_label(os, label, 0, 0);
}


} // end unnamed namespace


// ---------------- SBert wrapper ----------------
class SBertGGML {
    bert_ctx * ctx;
    int dim;
    int max_tokens;
public:
    SBertGGML(const std::string& model_path) {
        ctx = bert_load_from_file(model_path.c_str());
        if (!ctx) throw std::runtime_error("Failed to load model");
        dim = bert_n_embd(ctx);
        max_tokens = bert_n_max_tokens(ctx);
#if STANDALONE
        std::cout << "Loaded SBERT GGML model. dim=" << dim << " max_tokens=" << max_tokens << "\n";
#else
	message_log (LOG_INFO, "Loaded SBERT GGML model '%s'. dim=%d max_tokens=%d",
		model_path.c_str(), dim, max_tokens);
#endif
    }
    ~SBertGGML(){ if(ctx) bert_free(ctx); }
    int embedding_dim() const { return dim; }
    int embedding_capacity() const { return max_tokens; }
    bert_ctx* raw() const { return ctx; }

    std::vector<float> encode_text(const std::string &text, bool debug=false) const {
#if defined(EMBEDDINGS_LEGACY_EXPERIMENTAL_CODE)
        const int MAX_TOKENS = 512;
        bert_vocab_id tokens[MAX_TOKENS];
#else
        bert_vocab_id tokens[max_tokens];
#endif
        int32_t n_tokens = 0;
        bert_tokenize(ctx, text.c_str(), tokens, &n_tokens, max_tokens);
        if (n_tokens <= 0) throw std::runtime_error("Tokenization failed");

        std::vector<float> emb((size_t)dim);
        bert_eval(ctx, 4, tokens, n_tokens, emb.data());

        double norm = 0.0;
        for (float v : emb) norm += (double)v * (double)v;
        norm = std::sqrt(norm);
        if (norm > 0.0) for (auto &v : emb) v = (float)(v / (float)norm);

        if (debug) std::cerr << "[DEBUG] encode_text(): n_tokens="<<n_tokens<<" norm="<<norm<<"\n";
        return emb;
    }
};

/*
 * `SBertGGML` is intentionally small: model lifetime + text embedding only.
 * Index/sharding/search policies live in separate classes below.
 */

// ---------------- SearchResult ----------------

struct SearchResult {
    float score;
    int64_t start;
    int64_t end;
};

// ---------------- Metric + config ----------------
enum class Metric { L2, InnerProduct, Cosine };

struct HnswConfig {
    std::string default_field = "default";
    std::string model =  "sbert.ggml";
    size_t max_elements = 100000;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 50;
    Metric metric = Metric::L2;
    //
    int max_tokens_per_chunk = 128; // Needs to be less than the max_tokens
    float overlap_percent = 0.1f;
    size_t knn_lookahead_scale = 5;
    //
    float  alpha = 0.8; // relative threshold
    size_t minN = 3; // always return at least 3
    float  gapDelta = 0.1; // treat a 0.1 gap as significant
    size_t adaptive_lookahead = 10 ; // check top 10 results for cluster drop-off
    //
    size_t relative_k = 500; // kNN for relative
    //
    size_t flush_threshold = 100;  // auto-flush after this many inserts/deletes
    // 
    // Use async search on shards when (3* number of cores/2) > number of shards. 
    // ideally we want 1 thread perhaps 2 per core. Here we specify 1.5 times.
    // Given how most machines these days have at least 2 or more cores and 3 shards
    // would already be pushing things on such a machine...
    bool search_async = true; // Only has a function when MT was defined true during compile
    unsigned int processor_count = 1; // This gets filled with the hint by OS

    //
    bool debug = false;
};

/*
 * HnswConfig stores both ANN construction knobs and query-time behavior.
 * In production, prefer persisting the subset that affects index compatibility
 * (model identity, metric semantics, embedding dimensions, HNSW parameters).
 */

// ---------------- Utility ----------------
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
# include <filesystem>
#elif defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
# include <unistd.h>
#endif
namespace {

inline bool file_exists(const std::string &p) {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
    return std::filesystem::exists(p); // At least C++16
#elif defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    return access(p.c_str(), F_OK) != -1;
#else
    std::ifstream f(p); return f.good();
#endif
}

/*
std::vector<size_t> findPositionsInRange(const std::string& filename, int64_t x) {
    std::vector<size_t> positions;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return positions;
    }
    
    int64_t first, second;
    size_t position = 0;
    
    while (file >> first >> second) {
        // Check if x is between first and second (inclusive)
        if ((first <= x && x <= second) || (second <= x && x <= first)) {
            positions.push_back(position);
        }
        position++;
    }
    
    file.close();
    return positions;
}
*/

} // end unamed namespace

// ---------------- BertIndex (single shard) ----------------
class BertIndex {
    SBertGGML &embedder;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    std::string sentences_path;
    std::string index_path;
    std::string offsets_path;
    HnswConfig cfg;
    size_t next_available_label;
    std::vector<size_t> recycled_labels;
    size_t pending_flush_changes = 0;

public:
    // Owns a single shard: sentence storage + offsets + one HNSW instance.
    BertIndex(SBertGGML &eb, const std::string &sentences, const std::string &idx, const std::string &offs, const HnswConfig &conf)
        : embedder(eb), sentences_path(sentences), index_path(idx), offsets_path(offs), cfg(conf), next_available_label(0)
    {
        int dim = embedder.embedding_dim();
        if (cfg.metric == Metric::L2) {
            space = std::make_unique<hnswlib::L2Space>(dim);
        } else {
            space = std::make_unique<hnswlib::InnerProductSpace>(dim);
        }
        // Need to make sure that the configured max chunk is less than max_tokens
        // anything greater would be truncated if we did not chunk it to at most max_tokens!
        int max_tokens = embedder.embedding_capacity(); 
        if (cfg.max_tokens_per_chunk > max_tokens) cfg.max_tokens_per_chunk = max_tokens;
    }

    ~BertIndex() {
        try {
            flush();  // ensure final save
        } catch (const std::exception &exception) {
            // don’t throw in destructor, just log
            std::cerr << "[WARN] Failed to save index on destruction: " << exception.what() << "\n";
        }
    }

    // Helper methods for merging shards
    // Return current size (max label used so far)
    size_t size() const { return next_available_label; }

    // Expose dimension and metric space for rebuilds
    hnswlib::SpaceInterface<float>* getSpace() const { return space.get(); }

    // Replace the HNSW index with a new one
    void replaceIndex(std::unique_ptr<hnswlib::HierarchicalNSW<float>> newIdx) {
       index = std::move(newIdx);
       pending_flush_changes++; // mark dirty since index changed
   }

   // Expose paths so ShardedIndex can clean up files
   const std::string& getSentencesPath() const { return sentences_path; }
   const std::string& getOffsetsPath()   const { return offsets_path; }
   const std::string& getIndexPath()     const { return index_path; }

   //


    // sync the in-memory HNSW index to disk.
   void flush() {
        // We only re-write if the index on disk and in-memory are different
        if (pending_flush_changes && index) index->saveIndex(index_path);
	pending_flush_changes = 0;
    }

    void buildIndexIfMissing() {
        if (file_exists(index_path) && file_exists(offsets_path)) {
            index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
            index->setEf((uint32_t)cfg.ef_search);
            std::ifstream fin(offsets_path, std::ios::binary | std::ios::ate);
            size_t fsize = (size_t)fin.tellg();
            size_t entries = fsize / 16;
            fin.seekg(0);
            for (size_t i = 0; i < entries; ++i) {
                int64_t s = read_int64(fin);
                int64_t e = read_int64(fin);
                if (s==0 && e==0) recycled_labels.push_back(i);
            }
            next_available_label = entries;
        } else {
            createEmptyIndexFiles();
        }
    }

    void createEmptyIndexFiles() {
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), cfg.max_elements, cfg.M, cfg.ef_construction);
        index->setEf((uint32_t)cfg.ef_search);
        { std::ofstream ofs(offsets_path, std::ios::binary | std::ios::trunc); }
        { std::ofstream sfs(sentences_path, std::ios::trunc); }
        next_available_label = 0;
        recycled_labels.clear();
        pending_flush_changes = 0;
    }

    size_t allocateLabel() {
        if (!recycled_labels.empty()) {
            size_t id = recycled_labels.back();
            recycled_labels.pop_back();
            // NOTE: We may want to set a flag to flush() after the new label is
            // used. We don't do this yet but may..
            return id;
        }
        if (next_available_label >= cfg.max_elements) throw std::runtime_error("Reached max_elements");
        return next_available_label++;
    }

    std::vector<std::vector<bert_vocab_id>> chunk_tokens(const std::string &sentence) const {
        const int MAX_TOKENS = embedder.embedding_capacity();
//      const int MAX_TOKENS = 512; // BERT is designed for typically 512 
        bert_vocab_id tokens[MAX_TOKENS];
        int32_t n_tokens = 0;
        bert_tokenize(embedder.raw(), sentence.c_str(), tokens, &n_tokens, MAX_TOKENS);

        if (n_tokens <= cfg.max_tokens_per_chunk)
            return { std::vector<bert_vocab_id>(tokens, tokens+n_tokens) };

        int stride = std::max(1, (int)(cfg.max_tokens_per_chunk * (1.0f - cfg.overlap_percent)));
        std::vector<std::vector<bert_vocab_id>> chunks;
        for (int i=0; i<n_tokens; i+=stride) {
            int end = std::min(i+cfg.max_tokens_per_chunk, n_tokens);
            chunks.emplace_back(tokens+i, tokens+end);
            if (end == n_tokens) break;
        }
        return chunks;
    }


#if defined(EMBEDDINGS_LEGACY_EXPERIMENTAL_CODE) /* PRODUCTION */


    // For production we don't have a sentences file and the start, end are
    // re-Isearch addresses
    void append(const std::string &sentence, int64_t s, int64_t e) {
        auto chunks = chunk_tokens(sentence);
        int64_t start = s;
        int64_t end   = e;
 	size_t  off;

        std::fstream ofs(offsets_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!ofs) {
            ofs.open(offsets_path, std::ios::out | std::ios::binary); // create if missing
        }

        for (auto &chunk : chunks) {
            size_t label = allocateLabel();

	    off = 0;
            // sentence append
            std::string chunk_text(chunk.size(), '\0');
            for (size_t i=0;i<chunk.size();i++) {
                if (chunk_text[i] = (char)chunk[i]) != '\0'); // <-- if you’re writing raw tokens
		  off++;
            }
	    if (start + off < e) end = start + off; 
	    else end = e;

            // offsets append
	    write_offset_entry_for_label(ofs, label, start, end);

	    start = end + 1 - std::max(0, (int)(cfg.max_tokens_per_chunk * (1.0f - cfg.overlap_percent)));

            // embedding
            std::vector<float> emb(embedder.embedding_dim());
            bert_eval(embedder.raw(), 4, chunk.data(), (int32_t)chunk.size(), emb.data());

            double norm=0; for (float v:emb) norm+=v*v; norm=std::sqrt(norm);
            if (norm>0) for (auto &v:emb) v/=norm;

            index->addPoint(emb.data(), (hnswlib::labeltype)label);

            pending_flush_changes++;
            if (pending_flush_changes >= cfg.flush_threshold) {
                flush();
            }
        }
        ofs.close();
    }
#endif
#if 1

    void append(const std::string &sentence) {
        auto chunks = chunk_tokens(sentence);

        std::ofstream sfs(sentences_path, std::ios::binary | std::ios::app);
        std::fstream ofs(offsets_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!ofs) {
            ofs.open(offsets_path, std::ios::out | std::ios::binary); // create if missing
        }

        for (auto &chunk : chunks) {
            size_t label = allocateLabel();

            // sentence append
            sfs.seekp(0,std::ios::end);
            int64_t start = (int64_t)sfs.tellp();
            std::string chunk_text(chunk.size(), '\0');
            for (size_t i=0;i<chunk.size();i++) {
                chunk_text[i] = (char)chunk[i]; // <-- if you’re writing raw tokens
            }
            sfs.write(chunk_text.data(), chunk_text.size());
            sfs.put('\n');
            int64_t end = (int64_t)sfs.tellp();

            // offsets append
            ofs.seekp((std::streamoff)label * 16);
            write_int64(ofs, start);
            write_int64(ofs, end);

            // embedding
            std::vector<float> emb(embedder.embedding_dim());
            bert_eval(embedder.raw(), 4, chunk.data(), (int32_t)chunk.size(), emb.data());

            double norm=0; for (float v:emb) norm+=v*v; norm=std::sqrt(norm);
            if (norm>0) for (auto &v:emb) v/=norm;

            index->addPoint(emb.data(), (hnswlib::labeltype)label);

            pending_flush_changes++;
            if (pending_flush_changes >= cfg.flush_threshold) {
                flush();
            }
        }
        ofs.close();
        sfs.close();
    }

#else

    void append(const std::string &sentence) {
        auto chunks = chunk_tokens(sentence);
        for (auto &chunk : chunks) {
            size_t label = allocateLabel();

#if 1
            std::ofstream sfs(sentences_path, std::ios::binary | std::ios::app);
#else
            std::fstream sfs(sentences_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::app);
#endif
            sfs.seekp(0, std::ios::end);
            int64_t start = (int64_t)sfs.tellp();
            sfs.write(sentence.data(), sentence.size()); sfs.put('\n');
            int64_t end = (int64_t)sfs.tellp();
            sfs.close();

            std::fstream ofs(offsets_path, std::ios::in | std::ios::out | std::ios::binary);
            if (!ofs) ofs.open(offsets_path, std::ios::out | std::ios::binary);
            ofs.seekp((std::streamoff)label * 16);
            write_int64(ofs, start);
            write_int64(ofs, end);
            ofs.close();

            std::vector<float> emb(embedder.embedding_dim());
            bert_eval(embedder.raw(), 4, chunk.data(), (int32_t)chunk.size(), emb.data());

            double norm=0; for (float v:emb) norm+=v*v; norm=std::sqrt(norm);
            if (norm>0) for (auto &v:emb) v/=norm;

            index->addPoint(emb.data(), (hnswlib::labeltype)label);
            if (++pending_flush_changes > cfg.flush_threshold) flush();
        }
        // save(); // we defer this as its too expensve!
    }
#endif

#if 1

    // Mark deleted in the graph
    void markDelete(size_t label) {
        if (!index) throw std::runtime_error("Index not initialized (delete)");
        index->markDelete((hnswlib::labeltype)label);
        pending_flush_changes++;
    }

    // Remove the label from the graph and from the offsets.
    void remove(size_t label) {
        if (!index) throw std::runtime_error("Index not initialized (remove)");
        index->markDelete((hnswlib::labeltype)label);
        // zero offsets
        std::fstream ofs(offsets_path, std::ios::in | std::ios::out | std::ios::binary);
        ofs.seekp((std::streamoff)label * 16);
        write_int64(ofs, 0);
        write_int64(ofs, 0);
        ofs.close();
        recycled_labels.push_back(label);
        pending_flush_changes++;

        // We don't flush even if count as we want to defer
        // if (pending_flush_changes >= cfg.flush_threshold) flush();
    }

    // Can undelete but not unremove since the <start,end> was already zapped
    void undelete(size_t label) {
        if (!index) throw std::runtime_error("Index not initialized");
        index->unmarkDelete((hnswlib::labeltype)label);
        pending_flush_changes++;
        // if (pending_flush_changes >= cfg.flush_threshold) flush();
    }

   // Return the labels that have start,end range that contain address
    std::vector<size_t> labels_byAddress(int64_t address) {
        std::vector<size_t> matches;
        std::ifstream fin(offsets_path, std::ios::binary);
        fin.seekg(0, std::ios::end);
        size_t count = fin.tellg() / 16;
        fin.seekg(0);
        for (size_t i=0; i<count; i++) {
            int64_t s = read_int64(fin);
            int64_t e = read_int64(fin);
            if (address >= s && address < e) {
                matches.push_back(i);
            }
        }
        return matches;
    }

    // Return the labels that are included in the range <start,end>
    std::vector<size_t> labels_byAddress(int64_t start, int64_t end) {
        std::vector<size_t> matches;
        std::ifstream fin(offsets_path, std::ios::binary);
        fin.seekg(0, std::ios::end);
        size_t count = fin.tellg() / 16;
        fin.seekg(0);
        for (size_t i=0; i<count; i++) {
            int64_t s = read_int64(fin);
            int64_t e = read_int64(fin);
            if (s >= start && e <= end) {
                matches.push_back(i);
            }
        }
        return matches;
    }


    // Remove by address. Need to go through all the offsets.
    void remove_byAddress(int64_t gp) {
        auto matches = labels_byAddress(gp);
        for (auto lbl : matches) remove(lbl);
    }

    void delete_byAddress(int64_t gp) {
        auto matches = labels_byAddress(gp);
        for (auto lbl : matches) markDelete(lbl);
    }


    // Can undelete but can't unremove
    void undelete_byAddress(int64_t gp) {
        auto matches = labels_byAddress(gp);
        for (auto lbl : matches) undelete(lbl);
    }

#else

    void remove(size_t label) {
        if (!index) throw std::runtime_error("Index not initialized");

        // Mark as deleted in HNSWlib
        index->markDelete((hnswlib::labeltype)label);

        // Zero out offsets on disk
        std::fstream ofs(offsets_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open offsets file for remove()");
        ofs.seekp((std::streamoff)label * 16);
        write_int64(ofs, 0);
        write_int64(ofs, 0);
        ofs.close();

        // Make label reusable
        recycled_labels.push_back(label);

        pending_flush_changes++;
#if defined(EMBEDDINGS_LEGACY_EXPERIMENTAL_CODE) /* since we have 0,0 in the offset file and the search leaves it off we can defer */
        if (pending_flush_changes >= cfg.flush_threshold) flush();
#endif
    }
#endif

    float score_from_dist(float dist) const {
        if (cfg.metric == Metric::L2) return 1.0f/(1.0f+dist);
        return 1.0f - dist;
    }

    std::vector<SearchResult> search_knn(const std::vector<float> &query_embedding, size_t top_k) const {
        std::vector<SearchResult> search_results;
        int lookahead_count = (int)(top_k * cfg.knn_lookahead_scale);
        auto candidate_queue = index->searchKnn(query_embedding.data(), lookahead_count);
        std::vector<std::pair<float,hnswlib::labeltype>> candidate_pairs;
        while (!candidate_queue.empty()) { candidate_pairs.push_back(candidate_queue.top()); candidate_queue.pop(); }
        std::sort(candidate_pairs.begin(), candidate_pairs.end(), [](auto&a,auto&b){return a.first<b.first;});
        std::ifstream offsets_file(offsets_path, std::ios::binary);
        for (auto &candidate : candidate_pairs) {
            offsets_file.seekg((std::streamoff)candidate.second*16);
            const int64_t range_start = read_int64(offsets_file);
            const int64_t range_end = read_int64(offsets_file);
            if (range_start==0 && range_end==0) continue;
            search_results.push_back({score_from_dist(candidate.first), range_start, range_end});
            if (search_results.size()>=top_k) break;
        }
        return search_results;
    }

    std::vector<SearchResult> search_radius(const std::vector<float>&query_embedding,float minimum_score,size_t maxK=1000) const {
        auto candidate_results = search_knn(query_embedding,maxK);
        std::vector<SearchResult> radius_results;
        for(auto &result : candidate_results) if(result.score>=minimum_score) radius_results.push_back(result);
        return radius_results;
    }

    std::vector<SearchResult> search_relative(const std::vector<float>&query_embedding,float alpha = 0.0f){
        auto knn_results = search_knn(query_embedding, cfg.relative_k);
        if(knn_results.empty())return{};
        float effective_alpha = std::abs(alpha) > kFloatComparisonTolerance ? alpha : cfg.alpha;
        float score_cutoff = effective_alpha * knn_results.front().score;
        std::vector<SearchResult> filtered_results;
        for (auto &result : knn_results) if (result.score >= score_cutoff) filtered_results.push_back(result);
        return filtered_results;
    }

    std::vector<SearchResult> search_adaptive(const std::vector<float>&query_embedding,float alpha = 0.0f,
		size_t minN =0,size_t lookahead=0,float gapDelta = 0.0f){
        const float effective_alpha = std::abs(alpha) > kFloatComparisonTolerance ? alpha : cfg.alpha;
        const float effective_gap_delta = std::abs(gapDelta) > kFloatComparisonTolerance ? gapDelta : cfg.gapDelta;
        const size_t effective_lookahead = lookahead ? lookahead : cfg.adaptive_lookahead;
        const size_t minimum_results =  minN ? minN : cfg.minN;

        auto relative_results = search_relative(query_embedding, effective_alpha);
        if(relative_results.empty())return{};
        size_t stop_index = std::min(minimum_results, relative_results.size());
        for(size_t result_index=1; result_index < std::min(effective_lookahead, relative_results.size()); result_index++){
            float score_gap = relative_results[result_index-1].score - relative_results[result_index].score;
            if(score_gap >= effective_gap_delta){stop_index = std::max(minimum_results, result_index); break;}
            stop_index = std::max(stop_index, result_index + 1);
        }
        if(stop_index > relative_results.size()) stop_index = relative_results.size();
        return{relative_results.begin(), relative_results.begin()+stop_index};
    }


    void save() { flush(); /* We may also want to do something else here as well */ }

    std::string get_text(const SearchResult&result_range) const {
        std::ifstream sentence_file(sentences_path);
        sentence_file.seekg(result_range.start);
        std::string text((size_t)(result_range.end-result_range.start),'\0');
        sentence_file.read(&text[0], text.size());
        return text;
    }
};

// ---------------- ShardedIndex ----------------
class ShardedIndex {
    SBertGGML &embedder;
    std::string base_index_name;
    HnswConfig cfg;
    std::vector<std::unique_ptr<BertIndex>> shards;
    std::string shard_basename (size_t shard_index) const { return shard_index ? base_index_name+"_"+std::to_string(shard_index) : base_index_name;}
    std::string sentence_file_name(size_t shard_index) const { return shard_basename(shard_index)+kSentencesFileExtension;}
    std::string index_file_name(size_t shard_index) const {  return shard_basename(shard_index)+kIndexFileExtension;}
    std::string offsets_file_name(size_t shard_index) const { return shard_basename(shard_index)+kOffsetsFileExtension;}
public:
    // Coordinates shard creation, fan-out querying and shard-merge routines.
    ShardedIndex(SBertGGML &embedder_ref,const std::string &base_name,HnswConfig config)
        :embedder(embedder_ref),base_index_name(base_name),cfg(config){
        size_t shard_index=0;
#if MT
        // default is at least one CPU/Core.
        if (cfg.processor_count <= 1) cfg.processor_count = std::thread::hardware_concurrency();
#if STANDALONE
        if (cfg.debug) std::cout << "Cores: " << cfg.processor_count << "\n"; 
#endif
#endif
        while(file_exists(index_file_name(shard_index))&&file_exists(offsets_file_name(shard_index))){
            auto shard = std::make_unique<BertIndex>(embedder,sentence_file_name(shard_index),index_file_name(shard_index),offsets_file_name(shard_index),cfg);
            shard->buildIndexIfMissing();
            shards.push_back(std::move(shard));
            shard_index++;
        }
        if(shards.empty()) add_new();
    }
    void add_new(){
        size_t new_shard_index=shards.size();
        auto shard = std::make_unique<BertIndex>(embedder,sentence_file_name(new_shard_index),index_file_name(new_shard_index),offsets_file_name(new_shard_index),cfg);
        shard->buildIndexIfMissing();
        shards.push_back(std::move(shard));
    }
    void append(const std::string&text){
        try{shards.back()->append(text);}
        catch(std::runtime_error&exception){
            if(std::string(exception.what()).find("max_elements")!=std::string::npos){
                add_new(); shards.back()->append(text);
            } else throw;
        }
    }

#if 1

    void remove(size_t label, size_t shard=0) {
        if (shard >= shards.size()) throw std::runtime_error("Invalid shard index");
        shards[shard]->remove(label);
    }

    void markDelete(size_t label, size_t shard=0) {
        if (shard >= shards.size()) throw std::runtime_error("Invalid shard index");
        shards[shard]->markDelete(label);
    }

    void undelete(size_t label, size_t shard=0) {
        if (shard >= shards.size()) throw std::runtime_error("Invalid shard index");
        shards[shard]->undelete(label);
    }

    void delete_byAddress(int64_t address, size_t shard=0) {
        if (shard >= shards.size()) throw std::runtime_error("Invalid shard index");
        auto matches = shards[shard]->labels_byAddress(address);
        for (auto lbl : matches) shards[shard]->markDelete(lbl);
    }

    void remove_byAddress(int64_t address, size_t shard=0) {
        if (shard >= shards.size()) throw std::runtime_error("Invalid shard index");
        auto matches = shards[shard]->labels_byAddress(address);
        for (auto lbl : matches) shards[shard]->remove(lbl);
    }     

    void undelete_byAddress(int64_t address, size_t shard=0) {
        if (shard >= shards.size()) throw std::runtime_error("Invalid shard index");
        auto matches = shards[shard]->labels_byAddress(address);
        for (auto lbl : matches) shards[shard]->undelete(lbl);
    }

    size_t shard_count() const { return shards.size(); }

#endif

   void remove(size_t label){
        // for simplicity assume single shard for label mapping
        if(!shards.empty()) shards.back()->remove(label);
    }
    void flush() { for (auto &s : shards) { s->flush(); } }

#ifdef MT

/*
   std::vector<SearchResult> parallel_search(const std::string &query, size_t k) {
    std::vector<std::future<std::vector<SearchResult>>> futures;

    for (auto &shard : shards) {
        futures.push_back(std::async(std::launch::async, [&]() {
            return shard->knn(query, k); // per-shard search
        }));
    }

    std::vector<SearchResult> all;
    for (auto &f : futures) {
        auto partial = f.get();
        all.insert(all.end(), partial.begin(), partial.end());
    }

    // merge top-k results globally
    std::partial_sort(all.begin(), all.begin()+std::min(k,all.size()), all.end(),
                      [](auto &a, auto &b){ return a.score > b.score; });
    if (all.size() > k) all.resize(k);
    return all;
    }
*/

    template <typename SearchFunc> std::vector<SearchResult> parallel_search(SearchFunc search_fn, size_t topN = 0) {
        std::vector<std::future<std::vector<SearchResult>>> futures;

        for (auto &shard : shards) {
            futures.push_back(std::async(std::launch::async, [&]() {
                return search_fn(*shard);
            }));
        }

        std::vector<SearchResult> all;
        for (auto &f : futures) {
            auto partial = f.get();
            all.insert(all.end(), partial.begin(), partial.end());
        }

        if (topN > 0 && all.size() > topN) {
            std::partial_sort(all.begin(), all.begin() + topN, all.end(),
                          [](const SearchResult &a, const SearchResult &b) {
                              return a.score > b.score;
                          });
            all.resize(topN);
        } else {
            std::sort(all.begin(), all.end(),
                  [](const SearchResult &a, const SearchResult &b) {
                      return a.score > b.score;
                  });
        }

        return all;
    }


    // knn in a lamda
    std::vector<SearchResult> parallel_knn(const std::string &query_text, size_t top_k) {
        auto query_embedding = embedder.encode_text(query_text,cfg.debug);
        return parallel_search( [&](BertIndex &shard) { return shard.search_knn(query_embedding, top_k);  /* per-shard logic */ },
        top_k  /* global top-k */);
    }

    std::vector<SearchResult> parallel_radius(const std::string &query_text, float minimum_score) {
        auto query_embedding = embedder.encode_text(query_text,cfg.debug);
        return parallel_search( [&](BertIndex &shard) { return shard.search_radius(query_embedding, minimum_score); }) ;
    }

    std::vector<SearchResult> parallel_relative(const std::string&query_text,float alpha = 0.0f){
        auto query_embedding = embedder.encode_text(query_text,cfg.debug);
        float effective_alpha = std::abs(alpha) > kFloatComparisonTolerance ? alpha : cfg.alpha;
        return parallel_search( [&](BertIndex &shard) { return shard.search_relative(query_embedding, effective_alpha); }) ;
    }

    std::vector<SearchResult> parallel_adaptive(const std::string&query_text,
		float alpha = 0.0f,size_t minN =0,size_t lookahead=0,float gapDelta = 0.0f){
        const float effective_alpha = std::abs(alpha) > kFloatComparisonTolerance ? alpha : cfg.alpha;
        const float effective_gap = std::abs(gapDelta) > kFloatComparisonTolerance ? gapDelta : cfg.gapDelta;
        const size_t effective_lookahead = lookahead ? lookahead : cfg.adaptive_lookahead;
        const size_t effective_minimum_results =  minN ? minN : cfg.minN;
        auto query_embedding = embedder.encode_text(query_text,cfg.debug);

        return parallel_search( [&](BertIndex &shard) {
		return shard.search_adaptive(query_embedding, effective_alpha, effective_minimum_results, effective_lookahead, effective_gap); }) ;
    }
#endif

    std::vector<SearchResult> knn(const std::string&query_text,size_t top_k){
        auto query_embedding = embedder.encode_text(query_text,cfg.debug);
        std::vector<SearchResult>all_results;
        for(auto &shard : shards){auto shard_results = shard->search_knn(query_embedding,top_k);all_results.insert(all_results.end(),shard_results.begin(),shard_results.end());}
        std::sort(all_results.begin(),all_results.end(),[](auto&a,auto&b){return a.score>b.score;});
        if(all_results.size()>top_k)all_results.resize(top_k);return all_results;
    }
    std::vector<SearchResult> radius(const std::string&query_text,float minimum_score){
        auto query_embedding = embedder.encode_text(query_text,cfg.debug);
        std::vector<SearchResult>all_results;
        for(auto &shard : shards){auto shard_results = shard->search_radius(query_embedding,minimum_score);all_results.insert(all_results.end(),shard_results.begin(),shard_results.end());}
        std::sort(all_results.begin(),all_results.end(),[](auto&a,auto&b){return a.score>b.score;});
        return all_results;
    }

    std::vector<SearchResult> relative(const std::string&query_text,float alpha = 0.0f){
        auto knn_results = knn(query_text, cfg.relative_k);
        if(knn_results.empty())return{};
        float effective_alpha = std::abs(alpha) > kFloatComparisonTolerance ? alpha : cfg.alpha;
        float score_cutoff=effective_alpha*knn_results.front().score;
        std::vector<SearchResult>filtered_results;
        for(auto &result : knn_results) if(result.score>=score_cutoff) filtered_results.push_back(result);
        return filtered_results;
    }

    std::vector<SearchResult> adaptive(const std::string&query_text,float alpha = 0.0f,size_t minN =0,size_t lookahead=0,float gapDelta = 0.0f){
        const float effective_alpha = std::abs(alpha) > kFloatComparisonTolerance ? alpha : cfg.alpha;
        const float effective_gap = std::abs(gapDelta) > kFloatComparisonTolerance ? gapDelta : cfg.gapDelta;
        const size_t effective_lookahead = lookahead ? lookahead : cfg.adaptive_lookahead;
        const size_t effective_minimum_results =  minN ? minN : cfg.minN;

        auto relative_results = relative(query_text,effective_alpha);
        if(relative_results.empty())return{};
        size_t stop_index=std::min(effective_minimum_results,relative_results.size());
        for(size_t result_index=1;result_index<std::min(effective_lookahead,relative_results.size());result_index++){
            float score_gap=relative_results[result_index-1].score-relative_results[result_index].score;
            if(score_gap>=effective_gap){stop_index=std::max(effective_minimum_results,result_index);break;}
            stop_index=std::max(stop_index,result_index+1);
        }
        if(stop_index>relative_results.size())stop_index=relative_results.size();
        return{relative_results.begin(),relative_results.begin()+stop_index};
    }


    std::string get_text(const SearchResult&result_range){
        for(auto &shard : shards){
            auto shard_text = shard->get_text(result_range);
            if(!shard_text.empty()) return shard_text;
        }
        return "";
    }

    // We want to be able to merge the last two shards. This is maybe useful when the second shard
    // is relatively small and there is probably sufficient memory (and we're using a fast SSD for
    // swap) 
    bool merge_last_two() {
        if (shards.size() < 2) {
            return false; // throw std::runtime_error("Not enough shards to merge");
        }

        size_t first_shard_index = shards.size() - 2;
        size_t second_shard_index = shards.size() - 1;

        auto &destination_shard = shards[first_shard_index];
        auto &source_shard = shards[second_shard_index];

        // new capacity = combined size
        size_t merged_capacity = destination_shard->size() + source_shard->size();

        auto merged_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
		destination_shard->getSpace(), merged_capacity, cfg.M, cfg.ef_construction);

        // get base offset for sentences
        std::ifstream destination_sentences_reader(destination_shard->getSentencesPath(), std::ios::binary | std::ios::ate);
        int64_t destination_append_offset = (int64_t)destination_sentences_reader.tellg();
        destination_sentences_reader.close();

        std::ofstream destination_sentences_writer(destination_shard->getSentencesPath(), std::ios::app | std::ios::binary);
        std::fstream destination_offsets_file(destination_shard->getOffsetsPath(), std::ios::in | std::ios::out | std::ios::binary);

        std::ifstream source_sentences_file(source_shard->getSentencesPath(), std::ios::binary);
        std::ifstream source_offsets_file(source_shard->getOffsetsPath(), std::ios::binary);

        size_t source_offset_index = 0;
        while (source_offsets_file.peek() != EOF) {
            int64_t start=0, end=0;
            source_offsets_file.read((char*)&start,8);
            source_offsets_file.read((char*)&end,8);

            if (!source_offsets_file) break;

            if (start == 0 && end == 0) {
                // allocate but mark deleted
                size_t label = destination_shard->allocateLabel();
                std::vector<float> zero(embedder.embedding_dim(), 0.0f);
                merged_index->addPoint(zero.data(), (hnswlib::labeltype)label);
                merged_index->markDelete((hnswlib::labeltype)label);

                destination_offsets_file.seekp((std::streamoff)label * 16);
                write_int64(destination_offsets_file, 0);
                write_int64(destination_offsets_file, 0);
            } else {
                // read sentence
                std::string sentence(end - start, '\0');
                source_sentences_file.seekg(start);
                source_sentences_file.read(&sentence[0], sentence.size());

                // append to shard1 sentences
                destination_sentences_writer.write(sentence.data(), sentence.size());
                destination_sentences_writer.put('\n');

                int64_t newStart = destination_append_offset;
                int64_t newEnd   = destination_append_offset + sentence.size() + 1;
                destination_append_offset = newEnd;

                // update offsets
                size_t label = destination_shard->allocateLabel();
                destination_offsets_file.seekp((std::streamoff)label * 16);
                write_int64(destination_offsets_file, newStart);
                write_int64(destination_offsets_file, newEnd);

                // re-encode and add embedding
                auto emb = embedder.encode_text(sentence, cfg.debug);
                merged_index->addPoint(emb.data(), (hnswlib::labeltype)label);
            }
            source_offset_index++;
        }

        destination_offsets_file.close();
        destination_sentences_writer.close();

        // swap in new index
        destination_shard->replaceIndex(std::move(merged_index));

        // remove shard2 files
        std::remove(source_shard->getSentencesPath().c_str());
        std::remove(source_shard->getOffsetsPath().c_str());
        std::remove(source_shard->getIndexPath().c_str());

        // drop shard2
        shards.pop_back();

        return true; // OK
    }

   // Merge it all!!
   bool merge() {
     size_t count = 0;
     while (merge_last_two()) count++;
     return count ? true : false;
   }

};

// ---------------- Manager ----------------
class BertIndexManager {
    SBertGGML embedder;
    HnswConfig cfg;
    std::unordered_map<std::string,std::unique_ptr<ShardedIndex>> indices_by_name;
public:
    BertIndexManager(HnswConfig config) : BertIndexManager(config.model, config){};
    BertIndexManager(const std::string&model_path,HnswConfig config):embedder(model_path),cfg(config){}
    ShardedIndex&getOrCreateIndex(const std::string&index_name){
        if(!indices_by_name.count(index_name)) indices_by_name[index_name]=std::make_unique<ShardedIndex>(embedder,index_name,cfg);
        return*indices_by_name[index_name];
    }
#if 1 /* FOR TESTING */
    void append(const std::string&index_name,const std::string&text){getOrCreateIndex(index_name).append(text);}
#else
    // Production append
    void append(const std::string&index_name,const std::string&text, int64_t start, int64_t end){getOrCreateIndex(index_name).append(text, start, end);}
#endif

#if 1
    void remove(const std::string&index_name,size_t label,size_t shard=0){getOrCreateIndex(index_name).remove(label,shard);}
    void markDelete(const std::string&index_name,size_t label,size_t shard=0){getOrCreateIndex(index_name).markDelete(label,shard);}
    void undelete(const std::string&index_name,size_t label,size_t shard=0){getOrCreateIndex(index_name).undelete(label,shard);}
    void delete_byAddress(const std::string&index_name,int64_t address,size_t shard=0){getOrCreateIndex(index_name).delete_byAddress(address,shard);}
    void remove_byAddress(const std::string&index_name,int64_t address,size_t shard=0){getOrCreateIndex(index_name).remove_byAddress(address,shard);}
    void undelete_byAddress(const std::string&index_name,int64_t address,size_t shard=0){getOrCreateIndex(index_name).undelete_byAddress(address,shard);}
    size_t shard_count(const std::string&index_name){return getOrCreateIndex(index_name).shard_count();}
#else
    void remove(const std::string&index_name,size_t label){getOrCreateIndex(index_name).remove(label);}
#endif
    void flush(const std::string&index_name){getOrCreateIndex(index_name).flush();}

    std::vector<SearchResult> knn(const std::string&index_name,const std::string&query_text,size_t top_k=5){return getOrCreateIndex(index_name).knn(query_text,top_k);}
    std::vector<SearchResult> radius(const std::string&index_name,const std::string&query_text,float minimum_score = 0.0){
        return getOrCreateIndex(index_name).radius(query_text,minimum_score);
    }
    std::vector<SearchResult> relative(const std::string&index_name,const std::string&query_text,float alpha = 0.0){
        return getOrCreateIndex(index_name).relative(query_text,alpha);
    }
    std::vector<SearchResult> adaptive(const std::string&index_name,const std::string&query_text,
	float alpha = 0.0,size_t minN = 0,size_t lookahead=0,float gapDelta=0){
       return getOrCreateIndex(index_name).adaptive(query_text,alpha,minN,lookahead,gapDelta);
    }
    std::string text(const std::string&index_name,const SearchResult&result_range){return getOrCreateIndex(index_name).get_text(result_range);}

    void merge(const std::string& index_name) { getOrCreateIndex(index_name).merge(); }
    bool merge_last_two(const std::string& index_name) { return getOrCreateIndex(index_name).merge_last_two(); }
};

#if 1

enum class SearchType { Adaptive, Knn, Radius, Relative };

class EmbeddingIndexer
{
   BertIndexManager         *index_manager;
   HnswConfig                config;
   SearchType                search_mode;
   std::vector<SearchResult> empty;

   bool initialize_manager() {
     if (index_manager || (file_exists(config.model) && (index_manager = new BertIndexManager(config)) != NULL))
	return true;
     return false;
   }
public:
   // db_hnsw, db_nsg, db_IVFFlat
   EmbeddingIndexer(HnswConfig& cfg) : index_manager(NULL), config(cfg), search_mode(SearchType::Knn) {
   };

  ~EmbeddingIndexer() {
    if (index_manager) delete index_manager;
  }
   bool setModelPath(const std::string& path) {
     if (file_exists(path)) {
       config.model = path;
       return true;
     }
     return false;
   }

   void flush(const std::string& fieldname) {
     // We flush all the types we have
     if (index_manager) index_manager->flush(fieldname); // Right now only Bert/HNSW
   }
   void merge_last_two(const std::string& fieldname) {
     flush(fieldname);
     if (index_manager)  index_manager->merge_last_two(fieldname);
   }
   void merge(const std::string& fieldname) {
     flush(fieldname);
     if (index_manager) index_manager->merge(fieldname);
   }
   std::vector<SearchResult>  search_adaptive(const std::string& fieldname, const std::string& query) {
     initialize_manager();
     return index_manager ?  index_manager->adaptive(fieldname, query) : empty;
   }
   std::vector<SearchResult>  search_knn(const std::string& fieldname, const std::string& query) {
     initialize_manager();
     return index_manager ?  index_manager->knn(fieldname, query) : empty;
   }
   std::vector<SearchResult>  search_relative(const std::string& fieldname, const std::string& query) {
     initialize_manager();
     return index_manager ?  index_manager->relative(fieldname, query) : empty;
   }
   std::vector<SearchResult>  search_radius(const std::string& fieldname, const std::string& query) {
     initialize_manager();
     return index_manager ?  index_manager->radius(fieldname, query) : empty;
   }    

   std::vector<SearchResult>  adaptive_search(const std::string& fieldname, const std::string& query) {
     if (query.empty()) return empty;
     std::string index = (fieldname.empty()) ? config.default_field : fieldname;
     switch(search_mode) {
        case SearchType::Adaptive:  return search_adaptive(index, query);
        case SearchType::Relative:  return search_relative(index, query);
        case SearchType::Radius:  return search_radius(index, query);
        case SearchType::Knn: default:  return search_knn(index, query);
     }
   }

   bool append(const std::string& buffer, const std::string& fieldname, int64_t start, int64_t end, int type) {
       switch (type) {
	case 0: initialize_manager();
#if defined(EMBEDDINGS_LEGACY_EXPERIMENTAL_CODE)
	// Production code
	if (index_manager) {
	  index_manager->append(buffer, fieldname, start, end);
	  return true;
	}
	break;
#endif
	default: break;
     }
     return false;
   }

} ;


#endif

#if STANDALONE

// ---------------- Main ----------------
int main(int argc,char**argv){
   if(argc<2){
usage:
        std::cerr<<"Usage: "<<argv[0]<<" <sbert.ggml> [--metric l2|ip|cos] [--chunk max_tokens overlap] [--debug]\n";
        return 1;
    }   

    HnswConfig cfg; 

    Metric chosen=Metric::L2; // Metric::Cosine
    for(int i=2;i<argc;i++){
        std::string arg=argv[i];
        if(arg=="--metric" && i+1<argc){
            std::string val=argv[++i];
            if(val=="l2") chosen=Metric::L2;
            else if(val=="ip") chosen=Metric::InnerProduct;
            else if(val=="cos") chosen=Metric::Cosine;
            else std::cerr << "Unknown metric: " << val << "\n";
        } else if(arg=="--debug" || arg == "-d"){
            cfg.debug=true;
        } else if(arg=="--chunk" && i+2<argc){
            // Need to look at max_seq_length of the sBert
            if (( cfg.max_tokens_per_chunk=std::stoi(argv[++i])) > 512)
              std::cerr << "Warning: large chunk size specified (normally at most 128-512 tokens)\n";
            auto overlap = std::stof(argv[++i]);
            if (overlap > 0.1f && overlap < 1.0f) cfg.overlap_percent = overlap;
            else if (overlap < 100) cfg.overlap_percent=overlap/100.0f;
            else {
               std::cerr << "Absurd overlap specified: " << overlap << "% (recomended is 10-20)\n";
               return -1;
            }
        }
    }

    cfg.metric=chosen;
    std::cout<<"Using metric: "<<(cfg.metric==Metric::L2?"L2":cfg.metric==Metric::InnerProduct?"InnerProduct":"Cosine")<<"\n";
    if(cfg.debug) std::cout<<"Debug mode ON\n";

    BertIndexManager index_manager(argv[1],cfg);
    std::string current_index_name="default";
    std::cout<<"Commands: append <txt>, knn <k> <q>, radius <minScore> <q>, relative <alpha> <q>, adaptive <alpha> <minN> <lookahead> <gapDelta> <q>, merge, quit\n";
    for(std::string command_line;std::cout<<"["<<current_index_name<<"]> ",std::getline(std::cin,command_line);){
        if(command_line=="quit")break;
        if(command_line.rfind("append ",0)==0){index_manager.append(current_index_name,command_line.substr(7));continue;}
        if(command_line.rfind("knn ",0)==0){
            std::istringstream command_stream(command_line.substr(4));
            size_t top_k; command_stream>>top_k;
            std::string query_text; std::getline(command_stream,query_text); if(!query_text.empty()&&query_text[0]==' ')query_text=query_text.substr(1);
            for(auto &result : index_manager.knn(current_index_name,query_text,top_k))std::cout<<" - [score="<<result.score<<"] "<<index_manager.text(current_index_name,result)<<"\n";
            continue;
        }
        if(command_line.rfind("radius ",0)==0){
            std::istringstream command_stream(command_line.substr(7));
            float minimum_score; command_stream>>minimum_score;
            std::string query_text; std::getline(command_stream,query_text); if(!query_text.empty()&&query_text[0]==' ')query_text=query_text.substr(1);
            for(auto &result : index_manager.radius(current_index_name,query_text,minimum_score))std::cout<<" - [score="<<result.score<<"] "<<index_manager.text(current_index_name,result)<<"\n";
            continue;
        }
        if(command_line.rfind("relative ",0)==0){
            std::istringstream command_stream(command_line.substr(9));
            float alpha_threshold; command_stream>>alpha_threshold;
            std::string query_text; std::getline(command_stream,query_text); if(!query_text.empty()&&query_text[0]==' ')query_text=query_text.substr(1);
            for(auto &result : index_manager.relative(current_index_name,query_text,alpha_threshold))std::cout<<" - [score="<<result.score<<"] "<<index_manager.text(current_index_name,result)<<"\n";
            continue;
        }
        if(command_line.rfind("adaptive ",0)==0){
            std::istringstream command_stream(command_line.substr(9));
            float alpha_threshold, gap_delta; size_t min_results, lookahead_count;
            command_stream>>alpha_threshold>>min_results>>lookahead_count>>gap_delta;
            std::string query_text; std::getline(command_stream,query_text); if(!query_text.empty()&&query_text[0]==' ')query_text=query_text.substr(1);
            for(auto &result : index_manager.adaptive(current_index_name,query_text,alpha_threshold,min_results,lookahead_count,gap_delta))std::cout<<" - [score="<<result.score<<"] "<<index_manager.text(current_index_name,result)<<"\n";
            continue;
        }

        if (command_line == "merge") {
            if (index_manager.merge_last_two(current_index_name))
              std::cout << "Merged last two shards of index '" << current_index_name << "'.\n";
           else std::cout << "No two shards to merge in '" << current_index_name <<  "'.\n";
           continue;
        }

	// Assume a query
        for(auto &result : index_manager.knn(current_index_name,command_line,5))std::cout<<" - [score="<<result.score<<"] "<<index_manager.text(current_index_name,result)<<"\n";
            continue;

    }
}
#endif


/*
Notes & caveats

The code assumes the presence of bert.h with the C-style functions:
   bert_load_from_file, bert_tokenize, bert_eval, bert_free, bert_n_embd.
Keep those symbols available at link time.

A working bert is provided with this distribution.

hnswlib::HierarchicalNSW constructors and methods are used as in header-only hnswlib (common distribution).
If your hnswlib is built differently, adjust constructors accordingly (I used both file-based constructor
and parameter-based creation).

A working hnswlib is provided with this distribution

Cosine similarity is implemented by normalizing embeddings — HNSW uses inner-product space, so Cosine behaves
like InnerProduct on normalized vectors.

If you choose --metric cos, set metric to Cosine and the code normalizes. For L2 we use L2Space.

The offsets file stores pairs of 8-byte signed integers (start, end), so each entry is 16 bytes.
Deletion marks offsets as zero and calls markDelete(label) on the HNSW index; freed labels are reused.
When a shard reaches max_elements, a new shard is automatically created.
--debug prints query norms, chunk norms, raw distances and offsets to stderr.

Shards are only numbered after the first. Typically since we are creating a specific index per field
and sharding is only when we have more than (in the default) 100000 elements, we will probably have no
additional shards. NOTE: Performance across shards in linear..

Right now we use sentences and start/end but when its wrapped into re-Isearch we'll dump the sentences
as they are just for development and debugging and instead of start and end we'll store start GP and
end GP, basically a FC. Each start GP encodes not just the start of the field in the file but also the
identity of the file.

*/
