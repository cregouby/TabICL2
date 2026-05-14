# thanks to {torchvision} for the following
# download_and_cache <- function(url, redownload = FALSE, prefix = "TabICL2",  md5 = NULL, progress = TRUE) {
#
#   cache_path <- rappdirs::user_cache_dir("torch")
#
#   fs::dir_create(cache_path)
#   if (!is.null(prefix)) {
#     cache_path <- file.path(cache_path, prefix)
#   }
#   try(fs::dir_create(cache_path, recurse = TRUE), silent = TRUE)
#   path <- file.path(cache_path, fs::path_sanitize(fs::path_file(url)))
#
#   if (!file.exists(path) || redownload) {
#     # we should first download to a temporary file because
#     # download probalems could cause hard to debug errors.
#     tmp <- tempfile(fileext = fs::path_ext(path))
#     on.exit({try({fs::file_delete(tmp)}, silent = TRUE)}, add = TRUE)
#
#     withr::with_options(
#       list(timeout = max(600, getOption("timeout", default = 0))),
#       utils::download.file(url, tmp, mode = "wb")
#     )
#     fs::file_move(tmp, path)
#   } else {
#     if (!is.null(md5)) {
#       actual_md5 <- md5sum(path)
#       if (!identical(actual_md5, md5)) {
#         cli_warn(
#           "Cached file {.file {path}} has mismatched checksum. please rerun with {.cmd redownload=TRUE}."
#         )
#         file.remove(path)
#       }
#     }
#
#   }
#   path
# }

.download_and_cache <- function(url, redownload = FALSE, prefix = "TabICL2", md5 = NULL, size_hint = NULL,
                                cache_dir = NULL, progress = TRUE) {
  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  if (!is.null(prefix)) {
    cache_path <- file.path(cache_path, prefix)
  }
  try(fs::dir_create(cache_path, recurse = TRUE), silent = TRUE)
  dest_name <- fs::path_file(url)
  dest_path <- file.path(cache_path, fs::path_sanitize(dest_name))

  if (file.exists(dest_path)) {
    return(dest_path)
  }

  temp_dest <- paste0(dest_path, ".tmp")
  # on.exit({
  #   if (file.exists(temp_dest)) file.remove(temp_dest)
  # }, add = TRUE)

  size_msg <- if (!is.null(size_hint)) paste0(" (~", size_hint, ")") else ""
  cli_inform(
    "Downloading {.file {dest_name}}{size_msg} to {.file {cache_path}}"
  )

  if (progress) {
    download.file(url, destfile = temp_dest, mode = "wb")
  } else {
    download.file(url, destfile = temp_dest, mode = "wb", quiet = TRUE)
  }

  if (!file.exists(temp_dest) || file.info(temp_dest)$size == 0) {
    cli_abort("Download failed for {.url {url}}")
  }

  if (!is.null(md5)) {
    actual_md5 <- md5sum(temp_dest)
    if (!identical(actual_md5, md5)) {
      file.remove(temp_dest)
      cli_abort(
        "Corrupt file! Checksum mismatch for {.file {dest_name}}. Delete the file in {.file {cache_path}} and try again."
      )
    }
  }

  file.rename(temp_dest, dest_path)
  dest_path
}


.get_checkpoint_info <- function(checkpoint_version) {
  registry_names <- names(.model_urls)
  matched <- registry_names[grepl(checkpoint_version, registry_names, fixed = TRUE)]

  if (length(matched) == 0) {
    valid <- paste(registry_names, collapse = ", ")
    value_error(
      "Unknown checkpoint '{checkpoint_version}'. Valid options: {valid}."
    )
  }

  if (length(matched) > 1) {
    cli_warn(
      "Multiple checkpoints match '{checkpoint_version}'. Using the first: {.val {matched[1]}}"
    )
  }

  .model_urls[[matched[1]]]
}

.resolve_checkpoint_path <- function(model_path, checkpoint_version, allow_auto_download, progress = TRUE) {
  if (!is.null(model_path)) {
    if (file.exists(model_path)) {
      return(model_path)
    }
    if (!allow_auto_download) {
      value_error(
        "Checkpoint not found at {.file {model_path}} and allow_auto_download is FALSE."
      )
    }
    cli_inform(
      "File {.file {model_path}} not found. Attempting to download '{checkpoint_version}' from Hugging Face Hub."
    )
  }

  if (!allow_auto_download) {
    value_error(
      "Checkpoint '{checkpoint_version}' not found locally and automatic download is disabled."
    )
  }

  info <- .get_checkpoint_info(checkpoint_version)
  url <- info[1]
  md5 <- info[2]
  size <- info[3]

  .download_and_cache(
    url = url,
    md5 = md5,
    size_hint = size,
    progress = progress
  )

}
