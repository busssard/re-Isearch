/*
Copyright (c) 2020-21 Project re-Isearch and its contributors: See CONTRIBUTORS.
It is made available and licensed under the Apache 2.0 license: see LICENSE
*/
#include "db.hxx"
#include <errno.h>
#include <stdlib.h>
#include <fstream>
#include <malloc.h>
#include <unistd.h>

// Where do I need this for? I don't know.
int errno;

// Default cache size in kilobytes.
// Maybe this should be an config option, just for easy testing and
//   determination for best system performance
// NOTE: page size is 1KB - do not change!!
#define CACHE_SIZE_IN_KB 64

/*
 * DB wrapper notes
 * ================
 *
 * This class is a thin adaptor over legacy Berkeley DB APIs used by the core
 * indexing/search components. It keeps cursor-centric operations explicit and
 * exposes a small, stable surface used throughout the codebase.
 *
 * Key invariants:
 *   - page size stays 1024 bytes for existing on-disk compatibility
 *   - `isOpen` gates all operations that touch `dbp` / `dbcp`
 *   - `Start_*` methods position cursors; `Get_*` methods advance iteration
 */

DB::DB()
{
  isOpen = false;
}


DB::~DB()
{
  if (isOpen)
    Close();
}


int DB::OpenReadWrite(const STRING& filename, int mode)
{
  // Open existing database read/write, or create it when absent.
  // Initialize the database environment.
  dbenv = db_init((char *)NULL);
  memset(&dbinfo, 0, sizeof(dbinfo));
// dbinfo.db_cachesize = CACHE_SIZE_IN_KB * 1024;	// Cachesize: 64K.
  dbinfo.db_pagesize = 1024;      			// Page size: 1K.

  // Create the database.
  if (access(filename, F_OK) == 0)
    errno = db_open(filename, DB_BTREE, 0, 0, dbenv, &dbinfo, &dbp);
  else
    errno = db_open(filename, DB_BTREE, DB_CREATE, mode, dbenv, &dbinfo, &dbp);
  if (errno == 0)
    {
      // Acquire a cursor for the database.
      if ((seqrc = dbp->cursor(dbp, NULL, &dbcp, 0)) != 0)
	{
          seqerr = seqrc;
	  isOpen = 0;
          Close();
	  return NOTOK;
        }
      Path = filename;
      isOpen = 1;
      return OK;
    }
  Path.Clear();
  return NOTOK;
}


int DB::OpenRead(const STRING& filename)
{
    // Open in strict read-only mode for safe query/inspection workflows.
    //
    // Initialize the database environment.
    //
    dbenv = db_init((char *)NULL);
    memset(&dbinfo, 0, sizeof(dbinfo));
//    dbinfo.db_cachesize = CACHE_SIZE_IN_KB * 1024;	// Cachesize: 64K.
    dbinfo.db_pagesize = 1024;				// Page size: 1K.

    //
    // Open the database.
    //
    if ((errno = db_open(filename, DB_BTREE, DB_RDONLY, 0, dbenv,
			 &dbinfo, &dbp)) == 0)
    {
        //
	// Acquire a cursor for the database.
	//
        if ((seqrc = dbp->cursor(dbp, NULL, &dbcp, 0)) != 0)
	{
            seqerr = seqrc;
	    isOpen = 0;
            Close();
	    return NOTOK;
        }
	isOpen = 1;
	return OK;
    }
    else
    {
	return NOTOK;
    }
}


int DB::Close()
{
    // Close cursor first, then DB handle, then environment.
    if (isOpen)
    {
	//
	// Close cursor, database and clean up environment
	//
        (void)(dbcp->c_close)(dbcp);
	(void)(dbp->close)(dbp, 0);
	(void) db_appexit(dbenv);
    }
    isOpen = 0;
    return OK;
}


void DB::Start_Get()
{
    // Position sequence cursor at first key in lexical order.
    DBT	next_key;

    //
    // skey and next_key are just dummies
    //
    memset(&next_key, 0, sizeof(DBT));
    memset(&skey, 0, sizeof(DBT));

//    skey.data = "";
//    skey.size = 0;
//    skey.flags = 0;
    if (isOpen && dbp)
    {
	//
	// Set the cursor to the first position.
	//
        seqrc = dbcp->c_get(dbcp, &skey, &next_key, DB_FIRST);
	seqerr = seqrc;
    }
}


STRING DB::Get_Next()
{
    // Return current key and advance cursor by one.
    //
    // Looks like get Get_Next() and Get_Next_Seq() are pretty much the same...
    //
    DBT	next_key;
	
    memset(&next_key, 0, sizeof(DBT));

    if (isOpen && !seqrc)
    {
	STRING current_key((char *)skey.data, 0, skey.size);
	skey.flags = 0;
	seqrc = dbcp->c_get(dbcp, &skey, &next_key, DB_NEXT);
	seqerr = seqrc;
	return current_key;
    }
    else
	return NulString;
}

void DB::Start_Seq(const STRING& Key)
{
    // Position cursor at first key >= requested prefix.
    DBT	next_key;

    memset(&skey, 0, sizeof(DBT));
    memset(&next_key, 0, sizeof(DBT));

    skey.data = Key.c_str();
    skey.size = Key.length();
    if (isOpen && dbp)
    {
	//
	// Okay, get the first key. Use DB_SET_RANGE for finding partial
	// keys also. If you set it to DB_SET, and the words book, books
	// and bookstore do exists, it will find them if you specify
	// book*. However if you specify boo* if will not find
	// anything. Setting to DB_SET_RANGE will still find the `first'
	// word after boo* (which is book).
	//
        seqrc = dbcp->c_get(dbcp, &skey, &next_key, DB_SET_RANGE);
	seqerr = seqrc;
    }
}


STRING DB::Get_Next_Seq()
{
    // Continue sequence started by Start_Seq.
    DBT	next_key;
	
    memset(&next_key, 0, sizeof(DBT));

    if (isOpen && !seqrc)
    {
        STRING current_key((char *)skey.data, 0, skey.size);

	skey.flags = 0;
        seqrc = dbcp->c_get(dbcp, &skey, &next_key, DB_NEXT);
	seqerr = seqrc;
	return current_key;
    }
  return 0;
}

int DB::Put(const STRING &key, const STRING &data)
{
    // Upsert key/value payload into the BTREE store.
    DBT	key_record, data_record;

    memset(&key_record, 0, sizeof(DBT));
    memset(&data_record, 0, sizeof(DBT));

    if (!isOpen)
	return NOTOK;

    key_record.data = key.get();
    key_record.size = key.length();

    data_record.data = data.get();
    data_record.size = data.length();

    //
    // A 0 in the flags in put means replace, if you didn't specify DB_DUP
    // somewhere else...
    //
    return (dbp->put)(dbp, NULL, &key_record, &data_record, 0) == 0 ? OK : NOTOK;
}


int DB::Get(const STRING &key, STRING &data)
{
    DBT	key_record, data_record;

    memset(&key_record, 0, sizeof(DBT));
    memset(&data_record, 0, sizeof(DBT));

    key_record.data = key.get();
    key_record.size = key.length();

    int lookup_status = dbp->get(dbp, NULL, &key_record, &data_record, 0);
    if (lookup_status)
	return NOTOK;

    data = 0;
    data.append((char *)data_record.data, data_record.size);
    return OK;
}


int DB::Exists(const STRING &key)
{
    STRING data;

    if (!isOpen)
	return 0;

    return Get(key, data);
}


int DB::Delete(const STRING &key)
{
    DBT	k;

    memset(&k, 0, sizeof(DBT));

    if (!isOpen)
	return 0;

    k.data = key.get();
    k.size = key.length();

    return (dbp->del)(dbp, NULL, &k, 0);
}


DB *DB::getDatabaseInstance()
{
    return new DB();
}


/*
 * db_init --
 *      Initialize the environment. Only returns a pointer
 */
DB_ENV *DB::db_init(char *home)
{
  DB_ENV *dbenv;

  // Rely on calloc to initialize the structure.
  if ((dbenv = (DB_ENV *)calloc(sizeof(DB_ENV), 1)) == NULL)
    message_log (LOG_PANIC, "Insufficent Core!");
  else
    {
      dbenv->db_errfile = stderr;
      dbenv->db_errpfx = progname;
      if ((errno = db_appinit(home, NULL, dbenv, DB_CREATE)) != 0)
	{
	  message_log (LOG_PANIC, "db_appinit %s failed!", home.c_str())
	}
    }
    return (dbenv);
}
