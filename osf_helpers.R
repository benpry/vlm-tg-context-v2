## Helpers for listing files on an OSF node reliably.
##
## osfr::osf_ls_files() pages through the OSF API ten items at a time without a
## stable sort order, so on folders with many files it can return some files
## twice and skip others (seen with osfr 0.2.9 on the 96-file full_feedback
## folder: 3 duplicates, 3 missing). These helpers query the OSF v2 API directly,
## 100 items per page sorted by name, and follow the pagination links, so every
## file is listed exactly once. Every request failure stops with an error.

library(tidyverse)
library(httr)

OSF_API <- "https://api.osf.io/v2"

# GET a JSON document from the OSF API. Rate limiting (429) and server errors
# (5xx) are retried a few times with increasing waits; anything else, or a
# request that still fails after the retries, is an error.
osf_api_get <- function(url, max_attempts = 4) {
  headers <- if (nzchar(Sys.getenv("OSF_PAT"))) {
    add_headers(Authorization = paste("Bearer", Sys.getenv("OSF_PAT")))
  } else {
    add_headers()
  }
  for (attempt in seq_len(max_attempts)) {
    response <- GET(url, headers)
    status <- status_code(response)
    if (!http_error(response)) {
      return(content(response, as = "parsed", type = "application/json"))
    }
    retryable <- status == 429 || status >= 500
    if (!retryable || attempt == max_attempts) {
      stop("OSF API request failed (", status, ") after ", attempt, " attempt(s): ", url)
    }
    wait <- 10 * attempt
    message("OSF API returned ", status, "; retrying in ", wait, "s (attempt ", attempt, "/", max_attempts, ")")
    Sys.sleep(wait)
  }
}

# All items (files and folders) directly inside an osfstorage folder of a node,
# or at the top level of the node's osfstorage if folder_id is NULL.
osf_list_folder <- function(node_id, folder_id = NULL) {
  url <- str_c(OSF_API, "/nodes/", node_id, "/files/osfstorage/",
               if (is.null(folder_id)) "" else str_c(folder_id, "/"),
               "?page[size]=100&sort=name")
  items <- list()
  while (!is.null(url)) {
    page <- osf_api_get(url)
    items <- c(items, page$data)
    url <- page$links$`next`
  }
  if (length(items) == 0) {
    return(tibble(name = character(), kind = character(), id = character(),
                  md5 = character(), version = integer(), size = numeric(),
                  date_created = character(), date_modified = character()))
  }
  listing <- tibble(
    name = map_chr(items, \(i) i$attributes$name),
    kind = map_chr(items, \(i) i$attributes$kind),
    id = map_chr(items, \(i) i$id),
    md5 = map_chr(items, \(i) i$attributes$extra$hashes$md5 %||% NA_character_),
    version = map_int(items, \(i) as.integer(i$attributes$current_version %||% NA)),
    size = map_dbl(items, \(i) as.numeric(i$attributes$size %||% NA)),
    date_created = map_chr(items, \(i) i$attributes$date_created %||% NA_character_),
    date_modified = map_chr(items, \(i) i$attributes$date_modified %||% NA_character_)
  )
  if (anyDuplicated(listing$name)) {
    stop("OSF listing contains duplicate names: ",
         str_c(listing$name[duplicated(listing$name)], collapse = ", "))
  }
  listing
}
