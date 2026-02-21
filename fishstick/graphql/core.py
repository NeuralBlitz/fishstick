"""
Fishstick GraphQL Module - Comprehensive GraphQL implementation.

This module provides complete implementations for:
- GraphQL Server with schema creation
- Type System (ObjectType, InputType, InterfaceType, UnionType, EnumType)
- Resolvers (Resolver, FieldResolver, ObjectResolver)
- Mutations (create, update, delete)
- Subscriptions with PubSub
- Validation (query and schema)
- Execution (query and mutation)
- Utilities (GraphQL client)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
import asyncio
import hashlib
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class GraphQLError(Exception):
    """Base exception for GraphQL errors."""

    pass


class SchemaError(GraphQLError):
    """Raised when schema operation fails."""

    pass


class ResolverError(GraphQLError):
    """Raised when resolver operation fails."""

    pass


class ValidationError(GraphQLError):
    """Raised when validation fails."""

    pass


class ExecutionError(GraphQLError):
    """Raised when execution fails."""

    pass


class SubscriptionError(GraphQLError):
    """Raised when subscription operation fails."""

    pass


# =============================================================================
# Enums
# =============================================================================


class GraphQLTypeKind(Enum):
    """GraphQL type kind enumeration."""

    SCALAR = "SCALAR"
    OBJECT = "OBJECT"
    INTERFACE = "INTERFACE"
    UNION = "UNION"
    ENUM = "ENUM"
    INPUT_OBJECT = "INPUT_OBJECT"
    LIST = "LIST"
    NON_NULL = "NON_NULL"


class GraphQLKind(Enum):
    """GraphQL kind enumeration."""

    QUERY = "QUERY"
    MUTATION = "MUTATION"
    SUBSCRIPTION = "SUBSCRIPTION"


# =============================================================================
# GraphQL Server
# =============================================================================


@dataclass
class GraphQLServer:
    """GraphQL Server implementation."""

    schema: Optional["Schema"] = None
    resolvers: Dict[str, Callable] = field(default_factory=dict)
    mutations: Dict[str, Callable] = field(default_factory=dict)
    subscriptions: Dict[str, Callable] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    executor: Optional[ThreadPoolExecutor] = None

    def create_schema(self) -> "Schema":
        """Create a new GraphQL schema."""
        self.schema = Schema()
        return self.schema

    def add_query(self, name: str, resolver: Callable) -> "Query":
        """Add a query to the schema."""
        query = Query(name=name, resolver=resolver)
        self.resolvers[name] = resolver
        return query

    def add_mutation(self, name: str, resolver: Callable) -> "Mutation":
        """Add a mutation to the schema."""
        mutation = Mutation(name=name, resolver=resolver)
        self.mutations[name] = resolver
        return mutation

    def add_subscription(self, name: str, subscribe: Callable) -> "Subscription":
        """Add a subscription to the schema."""
        subscription = Subscription(name=name, subscribe=subscribe)
        self.subscriptions[name] = subscribe
        return subscription

    def execute(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        return execute_query(self, query, variables)

    def execute_mutation(
        self, mutation: str, variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a GraphQL mutation."""
        return execute_mutation(self, mutation, variables)


def graphql_server(**kwargs) -> GraphQLServer:
    """Create a GraphQL server instance."""
    return GraphQLServer(**kwargs)


# =============================================================================
# Schema Types
# =============================================================================


@dataclass
class GraphQLType:
    """Base class for GraphQL types."""

    name: str
    description: Optional[str] = None
    kind: GraphQLTypeKind = GraphQLTypeKind.SCALAR


@dataclass
class ScalarType(GraphQLType):
    """GraphQL Scalar type."""

    serialize: Optional[Callable] = None
    parse_value: Optional[Callable] = None
    parse_literal: Optional[Callable] = None

    def __post_init__(self):
        self.kind = GraphQLTypeKind.SCALAR


@dataclass
class ObjectType(GraphQLType):
    """GraphQL Object type."""

    fields: Dict[str, "Field"] = field(default_factory=dict)
    interfaces: List["InterfaceType"] = field(default_factory=list)

    def __post_init__(self):
        self.kind = GraphQLTypeKind.OBJECT

    def add_field(self, name: str, field_obj: "Field") -> "ObjectType":
        """Add a field to the object type."""
        self.fields[name] = field_obj
        return self

    def field(
        self,
        name: str,
        type_: str,
        resolver: Optional[Callable] = None,
        args: Optional[Dict] = None,
        description: Optional[str] = None,
    ) -> "ObjectType":
        """Add a field to the object type (decorator style)."""
        field_obj = Field(
            name=name,
            type=type_,
            resolver=resolver,
            args=args or {},
            description=description,
        )
        self.fields[name] = field_obj
        return self


Type = ObjectType


@dataclass
class InputType(GraphQLType):
    """GraphQL Input type."""

    fields: Dict[str, "InputField"] = field(default_factory=dict)

    def __post_init__(self):
        self.kind = GraphQLTypeKind.INPUT_OBJECT

    def add_field(
        self, name: str, type_: str, default_value: Any = None
    ) -> "InputType":
        """Add a field to the input type."""
        self.fields[name] = InputField(
            name=name, type=type_, default_value=default_value
        )
        return self


Input = InputType


@dataclass
class InterfaceType(GraphQLType):
    """GraphQL Interface type."""

    fields: Dict[str, "Field"] = field(default_factory=dict)
    implementing_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.kind = GraphQLTypeKind.INTERFACE

    def add_field(self, name: str, field_obj: "Field") -> "InterfaceType":
        """Add a field to the interface."""
        self.fields[name] = field_obj
        return self


Interface = InterfaceType


@dataclass
class UnionType(GraphQLType):
    """GraphQL Union type."""

    types: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.kind = GraphQLTypeKind.UNION

    def add_type(self, type_name: str) -> "UnionType":
        """Add a type to the union."""
        self.types.append(type_name)
        return self


Union = UnionType


@dataclass
class EnumValue:
    """GraphQL Enum value."""

    name: str
    value: Any
    description: Optional[str] = None


@dataclass
class EnumType(GraphQLType):
    """GraphQL Enum type."""

    values: Dict[str, EnumValue] = field(default_factory=dict)

    def __post_init__(self):
        self.kind = GraphQLTypeKind.ENUM

    def add_value(
        self, name: str, value: Any, description: Optional[str] = None
    ) -> "EnumType":
        """Add a value to the enum."""
        self.values[name] = EnumValue(name=name, value=value, description=description)
        return self


Enum = EnumType


# =============================================================================
# Fields
# =============================================================================


@dataclass
class Field:
    """GraphQL Field."""

    name: str
    type: str
    resolver: Optional[Callable] = None
    args: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    deprecation_reason: Optional[str] = None


Field = Field


@dataclass
class InputField:
    """GraphQL Input Field."""

    name: str
    type: str
    default_value: Any = None
    description: Optional[str] = None


# =============================================================================
# Schema
# =============================================================================


@dataclass
class Schema:
    """GraphQL Schema."""

    query_type: Optional[ObjectType] = None
    mutation_type: Optional[ObjectType] = None
    subscription_type: Optional[ObjectType] = None
    types: Dict[str, GraphQLType] = field(default_factory=dict)
    directives: List[Any] = field(default_factory=list)

    def query(self, name: str) -> ObjectType:
        """Set the query type."""
        self.query_type = ObjectType(name=name)
        return self.query_type

    def mutation(self, name: str) -> ObjectType:
        """Set the mutation type."""
        self.mutation_type = ObjectType(name=name)
        return self.mutation_type

    def subscription(self, name: str) -> ObjectType:
        """Set the subscription type."""
        self.subscription_type = ObjectType(name=name)
        return self.subscription_type

    def type(self, type_obj: GraphQLType) -> "Schema":
        """Add a type to the schema."""
        self.types[type_obj.name] = type_obj
        return self

    def add_type(self, name: str, type_obj: GraphQLType) -> "Schema":
        """Add a type to the schema."""
        self.types[name] = type_obj
        return self


# =============================================================================
# Resolvers
# =============================================================================


class Resolver(ABC):
    """Base resolver class."""

    def __init__(self, name: str, resolver: Optional[Callable] = None):
        self.name = name
        self.resolver = resolver

    @abstractmethod
    def resolve(self, info: "ResolveInfo", **kwargs) -> Any:
        """Resolve the field."""
        pass


@dataclass
class ResolveInfo:
    """Resolve info passed to resolvers."""

    field_name: str
    field_nodes: List[Any] = field(default_factory=list)
    return_type: Optional[GraphQLType] = None
    parent_type: Optional[GraphQLType] = None
    path: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[Schema] = None
    fragments: Dict[str, Any] = field(default_factory=dict)
    root_value: Any = None
    operation: Any = None
    variable_values: Dict[str, Any] = field(default_factory=dict)


class FieldResolver(Resolver):
    """Field resolver for object types."""

    def __init__(
        self,
        name: str,
        resolver: Optional[Callable] = None,
        type_: Optional[str] = None,
    ):
        super().__init__(name, resolver)
        self.type_ = type_

    def resolve(self, info: ResolveInfo, **kwargs) -> Any:
        """Resolve the field."""
        if self.resolver:
            return self.resolver(info, **kwargs)
        return None


Field = FieldResolver


class ObjectResolver(Resolver):
    """Object resolver for complex types."""

    def __init__(
        self,
        name: str,
        resolver: Optional[Callable] = None,
        fields: Optional[Dict[str, FieldResolver]] = None,
    ):
        super().__init__(name, resolver)
        self.fields = fields or {}

    def resolve(self, info: ResolveInfo, **kwargs) -> Any:
        """Resolve the object."""
        if self.resolver:
            return self.resolver(info, **kwargs)
        return {}

    def add_field(self, field: FieldResolver) -> "ObjectResolver":
        """Add a field resolver."""
        self.fields[field.name] = field
        return self


Object = ObjectResolver


def resolve_field(root: Any, info: ResolveInfo, **kwargs) -> Any:
    """Resolve a field value."""
    if hasattr(root, info.field_name):
        method = getattr(root, info.field_name)
        if callable(method):
            return method(**kwargs)
        return method
    if isinstance(root, dict):
        return root.get(info.field_name)
    return None


Resolve = resolve_field


# =============================================================================
# Mutations
# =============================================================================


@dataclass
class Mutation:
    """GraphQL Mutation."""

    name: str
    resolver: Callable
    description: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    type: Optional[str] = None

    def execute(self, info: ResolveInfo, **kwargs) -> Any:
        """Execute the mutation."""
        return self.resolver(info, **kwargs)


@dataclass
class MutationField:
    """GraphQL Mutation field."""

    name: str
    type: str
    resolver: Callable
    args: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


class CreateMutation(Mutation):
    """Create mutation."""

    def __init__(
        self,
        name: str,
        resolver: Callable,
        type_: str,
        description: Optional[str] = None,
    ):
        super().__init__(name, resolver, description)
        self.type_ = type_


class UpdateMutation(Mutation):
    """Update mutation."""

    def __init__(
        self,
        name: str,
        resolver: Callable,
        type_: str,
        description: Optional[str] = None,
    ):
        super().__init__(name, resolver, description)
        self.type_ = type_


class DeleteMutation(Mutation):
    """Delete mutation."""

    def __init__(
        self,
        name: str,
        resolver: Callable,
        type_: str,
        description: Optional[str] = None,
    ):
        super().__init__(name, resolver, description)
        self.type_ = type_


def create_mutation(
    name: str, resolver: Callable, type_: str, description: Optional[str] = None
) -> CreateMutation:
    """Create a create mutation."""
    return CreateMutation(name, resolver, type_, description)


def update_mutation(
    name: str, resolver: Callable, type_: str, description: Optional[str] = None
) -> UpdateMutation:
    """Create an update mutation."""
    return UpdateMutation(name, resolver, type_, description)


def delete_mutation(
    name: str, resolver: Callable, type_: str, description: Optional[str] = None
) -> DeleteMutation:
    """Create a delete mutation."""
    return DeleteMutation(name, resolver, type_, description)


Mutation = Mutation
Create = create_mutation
Update = update_mutation
Delete = delete_mutation


# =============================================================================
# Subscriptions
# =============================================================================


class Subscription(ABC):
    """GraphQL Subscription."""

    def __init__(
        self,
        name: str,
        subscribe: Optional[Callable] = None,
        resolve: Optional[Callable] = None,
    ):
        self.name = name
        self.subscribe = subscribe
        self.resolve = resolve

    @abstractmethod
    async def subscribe_async(
        self, info: "ResolveInfo", **kwargs
    ) -> AsyncGenerator[Any, None]:
        """Subscribe to events."""
        pass

    async def resolve_async(self, payload: Any, info: "ResolveInfo") -> Any:
        """Resolve the subscription event."""
        if self.resolve:
            return self.resolve(payload, info)
        return payload


@dataclass
class PubSub:
    """PubSub implementation for subscriptions."""

    channels: Dict[str, List[Callable]] = field(default_factory=dict)
    _events: asyncio.Queue = field(default_factory=asyncio.Queue)

    def publish(self, channel: str, payload: Any) -> None:
        """Publish to a channel."""
        if channel not in self.channels:
            self.channels[channel] = []
        for callback in self.channels[channel]:
            try:
                callback(payload)
            except Exception as e:
                logger.error(f"Error in pubsub callback: {e}")
        logger.info(f"Published to channel: {channel}")

    def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to a channel."""
        if channel not in self.channels:
            self.channels[channel] = []
        self.channels[channel].append(callback)

    def unsubscribe(self, channel: str, callback: Callable) -> None:
        """Unsubscribe from a channel."""
        if channel in self.channels:
            self.channels[channel].remove(callback)

    async def async_subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        """Async subscribe to a channel."""
        queue = asyncio.Queue()

        def callback(payload: Any):
            asyncio.create_task(queue.put(payload))

        self.subscribe(channel, callback)
        try:
            while True:
                payload = await queue.get()
                yield payload
        finally:
            self.unsubscribe(channel, callback)

    def get_subscribers(self, channel: str) -> List[Callable]:
        """Get subscribers for a channel."""
        return self.channels.get(channel, [])


async def subscribe(channel: str, pubsub: PubSub) -> AsyncGenerator[Any, None]:
    """Subscribe to a channel."""
    async for payload in pubsub.async_subscribe(channel):
        yield payload


Sub = Subscription
Publish = PubSub
Subscribe = subscribe


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationContext:
    """Validation context."""

    schema: Schema
    fragments: Dict[str, Any] = field(default_factory=dict)
    variable_values: Dict[str, Any] = field(default_factory=dict)
    errors: List[GraphQLError] = field(default_factory=list)


def validate_query(server: GraphQLServer, query: str) -> List[GraphQLError]:
    """Validate a GraphQL query."""
    errors: List[GraphQLError] = []

    if not query.strip():
        errors.append(ValidationError("Query cannot be empty"))
        return errors

    query_lower = query.strip().lower()

    if not query_lower.startswith(("query", "mutation", "subscription", "{")):
        errors.append(ValidationError("Invalid operation type"))

    braces_count = query.count("{") - query.count("}")
    if braces_count != 0:
        errors.append(ValidationError("Unmatched braces in query"))

    if "query" in query_lower and "(" in query:
        var_pattern = r"\$([a-zA-Z_][a-zA-Z0-9_]*)"
        variables = re.findall(var_pattern, query)
        if not variables and "$" in query:
            errors.append(ValidationError("Invalid variable syntax"))

    return errors


def validate_schema(schema: Schema) -> List[GraphQLError]:
    """Validate a GraphQL schema."""
    errors: List[GraphQLError] = []

    if schema.query_type is None:
        errors.append(SchemaError("Schema must have a query type"))

    for type_name, type_obj in schema.types.items():
        if isinstance(type_obj, ObjectType):
            for field_name, field_obj in type_obj.fields.items():
                if not field_obj.type:
                    errors.append(
                        SchemaError(
                            f"Field {field_name} in type {type_name} has no type"
                        )
                    )

    return errors


Query = validate_query
Schema = validate_schema


# =============================================================================
# Execution
# =============================================================================


def execute_query(
    server: GraphQLServer,
    query: str,
    variables: Optional[Dict] = None,
    operation_name: Optional[str] = None,
    root_value: Any = None,
) -> Dict[str, Any]:
    """Execute a GraphQL query."""
    errors = validate_query(server, query)
    if errors:
        return {"errors": [{"message": str(e)} for e in errors]}

    variables = variables or {}
    root = root_value or {}

    try:
        result = _execute_query_recursive(server, query, variables, root)
        return {"data": result}
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {"errors": [{"message": str(e)}]}


def _execute_query_recursive(
    server: GraphQLServer, query: str, variables: Dict, root: Any
) -> Any:
    """Recursively execute query parts."""
    query = query.strip()

    if query.startswith("{"):
        query = query[1:].strip()
    if query.endswith("}"):
        query = query[:-1].strip()

    result = {}
    current = root

    operation_match = re.match(
        r"(query|mutation|subscription)?\s*(\w+)?\s*(\([^)]*\))?\s*\{",
        query,
        re.IGNORECASE,
    )
    if operation_match:
        operation_type = operation_match.group(1) or "query"
        brace_pos = query.find("{")
        if brace_pos > 0:
            query = query[brace_pos + 1 :]
            if query.endswith("}"):
                query = query[:-1]

    fields = _parse_fields(query)

    for field_name, field_args in fields.items():
        resolver_key = field_name

        if resolver_key in server.resolvers:
            resolver = server.resolvers[resolver_key]
            args = _resolve_args(field_args, variables)

            try:
                if asyncio.iscoroutinefunction(resolver):
                    result[field_name] = asyncio.run(resolver(root, **args))
                else:
                    result[field_name] = resolver(root, **args)
            except Exception as e:
                logger.error(f"Resolver error for {field_name}: {e}")
                result[field_name] = None
        elif isinstance(current, dict):
            result[field_name] = current.get(field_name)
        elif hasattr(current, field_name):
            result[field_name] = getattr(current, field_name)
        else:
            result[field_name] = None

    return result


def _parse_fields(query: str) -> Dict[str, Dict]:
    """Parse query fields."""
    fields = {}
    current_field = ""
    brace_count = 0
    paren_count = 0
    in_string = False
    current_args = ""

    i = 0
    while i < len(query):
        char = query[i]

        if char == '"' and (i == 0 or query[i - 1] != "\\"):
            in_string = not in_string

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1

        if not in_string and paren_count == 0 and brace_count == 0:
            if char == " " and current_field and not current_args:
                if current_field.strip():
                    if current_field.strip() not in fields:
                        fields[current_field.strip()] = {}
                    current_field = ""
            elif char == "(" and current_field:
                current_args = ""
            elif char == ")" and current_args:
                if current_field.strip():
                    fields[current_field.strip()] = _parse_args(current_args)
                current_field = ""
                current_args = ""
            elif char == "," and not current_field:
                current_field = ""
            elif paren_count > 0:
                current_args += char
            elif brace_count == 0:
                current_field += char

        i += 1

    if current_field.strip():
        if current_field.strip() not in fields:
            fields[current_field.strip()] = {}

    return fields


def _parse_args(args_str: str) -> Dict[str, Any]:
    """Parse arguments string."""
    args = {}
    if not args_str.strip():
        return args

    pattern = r'(\w+):\s*("([^"]*)"|(\d+\.?\d*)|(\w+)|\[([^\]]*)\]|true|false|null)'
    matches = re.findall(pattern, args_str)

    for match in matches:
        name = match[0]
        value = match[1]

        if value.startswith('"') and value.endswith('"'):
            args[name] = value[1:-1]
        elif value == "true":
            args[name] = True
        elif value == "false":
            args[name] = False
        elif value == "null":
            args[name] = None
        elif "." in value:
            args[name] = float(value)
        elif value.isdigit():
            args[name] = int(value)
        else:
            args[name] = value

    return args


def _resolve_args(field_args: Dict, variables: Dict) -> Dict[str, Any]:
    """Resolve arguments including variables."""
    resolved = {}
    for key, value in field_args.items():
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            resolved[key] = variables.get(var_name)
        else:
            resolved[key] = value
    return resolved


def execute_mutation(
    server: GraphQLServer, mutation: str, variables: Optional[Dict] = None
) -> Dict[str, Any]:
    """Execute a GraphQL mutation."""
    errors = validate_query(server, mutation)
    if errors:
        return {"errors": [{"message": str(e)} for e in errors]}

    variables = variables or {}

    try:
        result = _execute_mutation_recursive(server, mutation, variables)
        return {"data": result}
    except Exception as e:
        logger.error(f"Mutation execution error: {e}")
        return {"errors": [{"message": str(e)}]}


def _execute_mutation_recursive(
    server: GraphQLServer, mutation: str, variables: Dict
) -> Any:
    """Recursively execute mutation parts."""
    mutation = mutation.strip()

    if mutation.startswith("mutation"):
        brace_pos = mutation.find("{")
        if brace_pos > 0:
            mutation = mutation[brace_pos + 1 :]
            if mutation.endswith("}"):
                mutation = mutation[:-1]

    fields = _parse_fields(mutation)
    result = {}

    for field_name, field_args in fields.items():
        if field_name in server.mutations:
            resolver = server.mutations[field_name]
            args = _resolve_args(field_args, variables)

            try:
                if asyncio.iscoroutinefunction(resolver):
                    result[field_name] = asyncio.run(resolver(**args))
                else:
                    result[field_name] = resolver(**args)
            except Exception as e:
                logger.error(f"Mutation error for {field_name}: {e}")
                result[field_name] = {"error": str(e)}
        else:
            result[field_name] = {"error": f"Mutation {field_name} not found"}

    return result


Execute = execute_query
Mutation = execute_mutation


# =============================================================================
# GraphQL Client
# =============================================================================


@dataclass
class GraphQLClient:
    """GraphQL Client for making requests."""

    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30

    async def query(
        self, query: str, variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        payload = {"query": query, "variables": variables or {}}

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url, json=payload, headers=self.headers, timeout=self.timeout
                ) as response:
                    return await response.json()
        except ImportError:
            logger.warning("aiohttp not available, using sync version")
            return self.query_sync(query, variables)
        except Exception as e:
            return {"errors": [{"message": str(e)}]}

    def query_sync(
        self, query: str, variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a GraphQL query synchronously."""
        payload = {"query": query, "variables": variables or {}}

        try:
            import requests

            response = requests.post(
                self.url, json=payload, headers=self.headers, timeout=self.timeout
            )
            return response.json()
        except ImportError:
            return {"errors": [{"message": "requests library not available"}]}
        except Exception as e:
            return {"errors": [{"message": str(e)}]}

    def mutate(self, mutation: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL mutation."""
        return self.query(mutation, variables)

    async def subscribe(
        self, subscription: str, callback: Callable[[Dict], None]
    ) -> None:
        """Subscribe to GraphQL subscriptions."""
        try:
            import websockets

            payload = {"query": subscription, "variables": {}}

            async with websockets.connect(self.url) as ws:
                await ws.send(json.dumps(payload))
                async for message in ws:
                    data = json.loads(message)
                    callback(data)
        except ImportError:
            logger.warning("websockets not available")
        except Exception as e:
            logger.error(f"Subscription error: {e}")


# =============================================================================
# Built-in Scalars
# =============================================================================


String = ScalarType(name="String", serialize=str, parse_value=str, parse_literal=str)
Int = ScalarType(name="Int", serialize=int, parse_value=int, parse_literal=int)
Float = ScalarType(
    name="Float", serialize=float, parse_value=float, parse_literal=float
)
Boolean = ScalarType(
    name="Boolean", serialize=bool, parse_value=bool, parse_literal=bool
)
ID = ScalarType(name="ID", serialize=str, parse_value=str, parse_literal=str)


# =============================================================================
# Directive Definitions
# =============================================================================


@dataclass
class Directive:
    """GraphQL Directive."""

    name: str
    description: Optional[str] = None
    locations: List[str] = field(default_factory=list)
    args: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Introspection Types
# =============================================================================


@dataclass
class IntrospectionSchema:
    """Introspection schema type."""

    queryType: Optional[Dict] = None
    mutationType: Optional[Dict] = None
    subscriptionType: Optional[Dict] = None
    types: List[Dict] = field(default_factory=list)
    directives: List[Dict] = field(default_factory=list)


def create_introspection_type(schema: Schema) -> ObjectType:
    """Create introspection type for schema."""
    return ObjectType(
        name="__Schema",
        description="A GraphQL Schema defines the capabilities of a GraphQL server",
        fields={
            "queryType": Field(name="queryType", type="__Type"),
            "mutationType": Field(name="mutationType", type="__Type"),
            "subscriptionType": Field(name="subscriptionType", type="__Type"),
            "types": Field(name="types", type="[__Type!]!"),
            "directives": Field(name="directives", type="[__Directive!]!"),
        },
    )


# =============================================================================
# Export Aliases
# =============================================================================


__all__ = [
    "GraphQLServer",
    "create_schema",
    "add_query",
    "add_mutation",
    "ObjectType",
    "InputType",
    "InterfaceType",
    "UnionType",
    "EnumType",
    "Type",
    "Input",
    "Interface",
    "Union",
    "Enum",
    "Resolver",
    "FieldResolver",
    "ObjectResolver",
    "resolve_field",
    "Resolve",
    "Field",
    "Object",
    "Mutation",
    "MutationField",
    "create_mutation",
    "update_mutation",
    "delete_mutation",
    "Create",
    "Update",
    "Delete",
    "Subscription",
    "PubSub",
    "subscribe",
    "Sub",
    "Publish",
    "Subscribe",
    "validate_query",
    "validate_schema",
    "Query",
    "Schema",
    "execute_query",
    "execute_mutation",
    "Execute",
    "Mutation as ExecuteMutation",
    "graphql_server",
    "GraphQLClient",
    "GraphQLType",
    "ScalarType",
    "Field",
    "InputField",
    "Schema",
    "Directive",
    "GraphQLError",
    "SchemaError",
    "ResolverError",
    "ValidationError",
    "ExecutionError",
    "SubscriptionError",
    "ResolveInfo",
    "GraphQLTypeKind",
    "GraphQLKind",
    "String",
    "Int",
    "Float",
    "Boolean",
    "ID",
]
