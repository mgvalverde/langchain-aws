from __future__ import annotations

import os
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import boto3
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import run_in_executor
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]


class BedrockRerank(BaseDocumentCompressor):
    """Bedrock chat model integration built on the Bedrock converse API.

    This implementation will eventually replace the existing ChatBedrock implementation
    once the Bedrock converse API has feature parity with older Bedrock API.
    Specifically the converse API does not yet support custom Bedrock models.

    Setup:
        To use Amazon Bedrock make sure you've gone through all the steps described
        here: https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html

        Once that's completed, install the LangChain integration:

        .. code-block:: bash

            pip install -U langchain-aws

    Key init args — completion params:
        model: str
            Name of BedrockConverse model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        region_name: Optional[str]
            AWS region to use, e.g. 'us-west-2'.
        base_url: Optional[str]
            Bedrock endpoint to use. Needed if you don't want to default to us-east-
            1 endpoint.
        credentials_profile_name: Optional[str]
            The name of the profile in the ~/.aws/credentials or ~/.aws/config files.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_aws import ChatBedrockConverse

            llm = ChatBedrockConverse(
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                temperature=0,
                max_tokens=None,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content=[{'type': 'text', 'text': "J'aime la programmation."}], response_metadata={'ResponseMetadata': {'RequestId': '9ef1e313-a4c1-4f79-b631-171f658d3c0e', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Sat, 15 Jun 2024 01:19:24 GMT', 'content-type': 'application/json', 'content-length': '205', 'connection': 'keep-alive', 'x-amzn-requestid': '9ef1e313-a4c1-4f79-b631-171f658d3c0e'}, 'RetryAttempts': 0}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': 609}}, id='run-754e152b-2b41-4784-9538-d40d71a5c3bc-0', usage_metadata={'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content=[], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'type': 'text', 'text': 'J', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': "'", 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': 'a', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': 'ime', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': ' la', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': ' programm', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': 'ation', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'text': '.', 'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[{'index': 0}], id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[], response_metadata={'stopReason': 'end_turn'}, id='run-da3c2606-4792-440a-ac66-72e0d1f6d117')
            AIMessageChunk(content=[], response_metadata={'metrics': {'latencyMs': 581}}, id='run-da3c2606-4792-440a-ac66-72e0d1f6d117', usage_metadata={'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36})

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content=[{'type': 'text', 'text': "J'aime la programmation.", 'index': 0}], response_metadata={'stopReason': 'end_turn', 'metrics': {'latencyMs': 554}}, id='run-56a5a5e0-de86-412b-9835-624652dc3539', usage_metadata={'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36})

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetWeather',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'tooluse_Mspi2igUTQygp-xbX6XGVw'},
             {'name': 'GetWeather',
              'args': {'location': 'New York, NY'},
              'id': 'tooluse_tOPHiDhvR2m0xF5_5tyqWg'},
             {'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'tooluse__gcY_klbSC-GqB-bF_pxNg'},
             {'name': 'GetPopulation',
              'args': {'location': 'New York, NY'},
              'id': 'tooluse_-1HSoGX0TQCSaIg7cdFy8Q'}]

        See ``ChatBedrockConverse.bind_tools()`` method for more.

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup='What do you call a cat that gets all dressed up?', punchline='A purrfessional!', rating=7)

        See ``ChatBedrockConverse.with_structured_output()`` for more.

    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            [{'type': 'text',
              'text': 'The image depicts a sunny day with a partly cloudy sky. The sky is a brilliant blue color with scattered white clouds drifting across. The lighting and cloud patterns suggest pleasant, mild weather conditions. The scene shows an open grassy field or meadow, indicating warm temperatures conducive for vegetation growth. Overall, the weather portrayed in this scenic outdoor image appears to be sunny with some clouds, likely representing a nice, comfortable day.'}]

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36}

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'ResponseMetadata': {'RequestId': '776a2a26-5946-45ae-859e-82dc5f12017c',
              'HTTPStatusCode': 200,
              'HTTPHeaders': {'date': 'Mon, 17 Jun 2024 01:37:05 GMT',
               'content-type': 'application/json',
               'content-length': '206',
               'connection': 'keep-alive',
               'x-amzn-requestid': '776a2a26-5946-45ae-859e-82dc5f12017c'},
              'RetryAttempts': 0},
             'stopReason': 'end_turn',
             'metrics': {'latencyMs': 1290}}
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_id: str = "cohere.rerank-v3-5:0"
    provider: str = "cohere"
    """Id of the model to call.

    e.g., ``"cohere.rerank-v3-5:0"``. This is equivalent to the 
    modelID property in the list-foundation-models api. For custom and provisioned 
    models, an ARN value is expected. See 
    https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns 
    for a list of all supported built-in models.
    """

    top_k: Optional[int] = 5
    include_additional_metadata_keys: List[str] = []

    """Max tokens to generate."""
    include_metadata_keys: Optional[List[str]] = None
    region_name: Optional[str] = None
    """The aws region, e.g., `us-west-2`. 

    Falls back to AWS_DEFAULT_REGION env variable or region specified in ~/.aws/config 
    in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = Field(default=None, exclude=True)
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files.

    Profile should either have access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used. 
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    aws_access_key_id: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id. 

    If provided, aws_secret_access_key must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.
    """

    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key. 

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.
    """

    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token. 

    If provided, aws_access_key_id and aws_secret_access_key must 
    also be provided. Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    endpoint_url: Optional[str] = Field(default=None, alias="base_url")
    """Needed if you don't want to default to us-east-1 endpoint"""

    config: Any = None
    """An optional botocore.config.Config instance to pass to the client."""

    additional_model_request_fields: Optional[Dict[str, Any]] = None
    """Additional inference parameters that the model supports.

    Parameters beyond the base set of inference parameters that Converse supports in the
    inferenceConfig field.
    """

    additional_model_response_field_paths: Optional[List[str]] = None
    """Additional model parameters field paths to return in the response. 

    Converse returns the requested fields as a JSON Pointer object in the 
    additionalModelResponseFields field. The following is example JSON for 
    additionalModelResponseFieldPaths.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that AWS credentials to and python package exists in environment."""
        # Skip creating new client if passed in constructor
        if self.client is None:
            creds = {
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
                "aws_session_token": self.aws_session_token,
            }
            if creds["aws_access_key_id"] and creds["aws_secret_access_key"]:
                session_params = {
                    k: v.get_secret_value() for k, v in creds.items() if v
                }
            elif any(creds.values()):
                raise ValueError(
                    f"If any of aws_access_key_id, aws_secret_access_key, or "
                    f"aws_session_token are specified then both aws_access_key_id and "
                    f"aws_secret_access_key must be specified. Only received "
                    f"{(k for k, v in creds.items() if v)}."
                )
            elif self.credentials_profile_name is not None:
                session_params = {"profile_name": self.credentials_profile_name}
            else:
                # use default credentials
                session_params = {}

            try:
                session = boto3.Session(**session_params)

                self.region_name = (
                        self.region_name
                        or os.getenv("AWS_DEFAULT_REGION")
                        or session.region_name
                )

                client_params = {
                    "endpoint_url": self.endpoint_url,
                    "config": self.config,
                    "region_name": self.region_name,
                }
                client_params = {k: v for k, v in client_params.items() if v}
                self.client = session.client("bedrock-agent-runtime", **client_params)
            except ValueError as e:
                raise ValueError(f"Error raised by bedrock service:\n\n{e}") from e
            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    f"profile name are valid. Bedrock error:\n\n{e}"
                ) from e

        return self

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "amazon_bedrock_reranker"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_aws", "reranker"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
        }

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.
        """
        if len(documents) == 0:
            return []
        text_payload = [build_payload(doc, include_keys=self.include_additional_metadata_keys) for doc in documents]
        model_package_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.model_id}"
        top_k = min(self.top_k, len(documents))
        response = self.client.rerank(
            queries=[
                {
                    "type": "TEXT",
                    "textQuery": {
                        "text": query
                    }
                }
            ],
            sources=text_payload,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": top_k,
                    "modelConfiguration": {
                        "modelArn": model_package_arn,
                    }
                }
            }
        )
        output = []
        for element in response['results']:
            idx = element['index']
            output.append(documents[idx])
        return output


def build_payload(document: Document, include_keys: Optional[List[str]] = None):
    page_content = document.page_content
    if include_keys:
        metadata = "\n".join(f"{k}: {v}" for k, v in document.metadata.items() if k in include_keys)
        page_content = f"""metadata: {metadata}\n----------------\n{page_content}"""

    return {
        'inlineDocumentSource': {
            'type': 'TEXT',
            'textDocument': {
                'text': page_content
            },
        },
        'type': 'INLINE'
    }
