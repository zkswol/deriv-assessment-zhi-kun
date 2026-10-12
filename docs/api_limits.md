# API usage

The API rate limit is 60 requests per minute per API token. Requests above this limit receive HTTP 429.

When HTTP 429 is returned, clients should wait for the Retry-After header duration before retrying. Use exponential backoff when that header is absent.
