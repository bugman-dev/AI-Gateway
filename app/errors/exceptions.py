from __future__ import annotations


class GatewayError(Exception):
    def __init__(self, message: str, *, code: str, error_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.error_type = error_type


class AuthenticationError(GatewayError):
    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(message, code="invalid_api_key", error_type="authentication_error")


class AuthorizationError(GatewayError):
    def __init__(self, message: str = "Not authorized to use this model") -> None:
        super().__init__(message, code="model_not_authorized", error_type="authorization_error")


class InvalidRequestError(GatewayError):
    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message, code=code, error_type="invalid_request_error")


class UnknownModelError(GatewayError):
    def __init__(self, logical_model: str) -> None:
        super().__init__(
            f"Unknown model '{logical_model}'",
            code="model_not_found",
            error_type="invalid_request_error",
        )
        self.logical_model = logical_model


class BackendUnavailableError(GatewayError):
    def __init__(self, message: str = "Inference backend is unavailable") -> None:
        super().__init__(message, code="backend_unavailable", error_type="backend_error")


class BackendError(GatewayError):
    def __init__(self, message: str = "Inference backend failed") -> None:
        super().__init__(message, code="backend_error", error_type="backend_error")
