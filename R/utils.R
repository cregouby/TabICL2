.model_urls <- list(
  # tabicl_classifier_v1_0 = c(
  #   "https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v1-20250208.ckpt",
  #   "TODO_MD5_V1", "85 MB"),
  # tabicl_classifier_v1_1 = c(
  #   "https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v1.1-20250506.ckpt",
  #   "TODO_MD5_V1_1", "87 MB"),
  tabicl_classifier_v2 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/tabicl2_classifier.safetensors",
    "5dd8f9d75f1512b08633c8b242e645c4", "105 MB"),
  tabicl_regressor_v2 = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/tabicl2_regressor.safetensors",
    "ffccabca67b4fd8ca61860e1195afc8b", "109 MB")
)

# deeply inspired from torchvision:::download_and_cache

#' @importFrom utils download.file
#' @importFrom fs path_file path_sanitize
#' @importFrom tools md5sum
.download_and_cache <- function(url, redownload = FALSE, prefix = "TabICL2", md5 = NULL, size_hint = NULL,
                                cache_dir = NULL, progress = TRUE) {
  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  if (!is.null(prefix)) {
    cache_path <- file.path(cache_path, prefix)
  }
  try(fs::dir_create(cache_path, recurse = TRUE), silent = TRUE)
  dest_name <- path_file(url)
  dest_path <- file.path(cache_path, path_sanitize(dest_name))

  if (!redownload && file.exists(dest_path)) {
    return(dest_path)
  }

  temp_dest <- paste0(dest_path, ".tmp")
  on.exit({
    if (file.exists(temp_dest)) file.remove(temp_dest)
  }, add = TRUE)

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
    if (!identical(actual_md5[[1L]], md5)) {
      cli_abort(
        "Corrupt file! Checksum mismatch for {.file {dest_name}}. Delete the file in {.file {cache_path}} and try again."
      )
    }
  }

  file.rename(temp_dest, dest_path)
  dest_path
}

# Resolve model_version to a local file path and load it as a torch object
#
# model_version may be:
#   - a registry key (partial or exact match in .model_urls)
#   - an https:// or http:// URL  -> downloaded and cached
#   - a file:// URI               -> stripped to a plain path and checked
#
# Returns the torch object
#' @importFrom torch torch_load load_state_dict
load_checkpoint_path <- function(model_version, progress = TRUE) {
  stopifnot(!is.null(model_version), is.character(model_version), length(model_version) == 1L)

  # file:// URI
  if (startsWith(model_version, "file://")) {
    path <- sub("^file://", "", model_version)
    if (!file.exists(path)) {
      cli_abort("Checkpoint not found at {.file {local_path}}.")
    }
  } else if (startsWith(model_version, "https://") || startsWith(model_version, "http://")) {
  # https:// or http:// URL
    path <- .download_and_cache(url = model_version, progress = progress)
  } else {
    # registry key
    registry_names <- names(.model_urls)
    matched <- registry_names[grepl(model_version, registry_names, fixed = TRUE)]
    if (length(matched) == 0L) {
      cli_abort(c(
        "{.val {model_version}} is not a known checkpoint key, a valid URL, or a {.code file://} path.",
        i = "Known registry keys: {.val {registry_names}}."
      ))
    }
    if (length(matched) > 1L) {
      cli_warn(
        "Multiple registry keys match {.val {model_version}}. Using {.val {matched[1L]}}."
      )
    } else {
      info <- .model_urls[[matched[1L]]]
      path <- .download_and_cache(
        url       = info[1L],
        md5       = info[2L],
        size_hint = info[3L],
        progress  = progress
      )
    }
  }


  if (fs::path_ext(path) == "safetensors") {
    return(torch_load(path))
  } else {
    return(load_state_dict(path))
  }
}

