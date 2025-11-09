package ma.emsi.kharraz;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class TestRoutage {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private static EmbeddingStore<TextSegment> createEmbeddingStore(String resourcePath) {
        Path documentPath = toPath(resourcePath);
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = loadDocument(documentPath, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddingModel.embedAll(segments).content(), segments);
        return embeddingStore;
    }

    public static void main(String[] args) {
        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-1.5-flash")
                .build();

        // Phase 1: Ingestion
        EmbeddingStore<TextSegment> ragStore = createEmbeddingStore("rag.pdf");

        // Phase 2: Retrieval
        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(ragStore)
                .embeddingModel(new AllMiniLmL6V2EmbeddingModel())
                .maxResults(2)
                .minScore(0.6)
                .build();

        class CustomQueryRouter implements QueryRouter {
            private final ChatModel chatModel;
            private final ContentRetriever retriever;

            CustomQueryRouter(ChatModel chatModel, ContentRetriever retriever) {
                this.chatModel = chatModel;
                this.retriever = retriever;
            }

            @Override
            public Collection<ContentRetriever> route(Query query) {
                PromptTemplate promptTemplate = PromptTemplate.from(
                        "Est-ce que la requête '{{query}}' porte sur l'IA ? Réponds seulement par 'oui', 'non' ou 'peut-être'."
                );
                Prompt prompt = promptTemplate.apply(Map.of("query", query.text()));

                Response<AiMessage> response = chatModel.generate(prompt.toMessages());
                String answer = response.content().text().trim().toLowerCase();

                System.out.println("Routing decision: Query is about AI? -> " + answer);

                if (answer.contains("non")) {
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(retriever);
                }
            }
        }

        QueryRouter customQueryRouter = new CustomQueryRouter(chatModel, ragRetriever);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(customQueryRouter)
                .build();

        // Create Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // Start chatting
        System.out.println("Assistant is ready. Ask your questions.");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("User: ");
            String userMessage = scanner.nextLine();

            if ("exit".equalsIgnoreCase(userMessage)) {
                break;
            }

            String assistantResponse = assistant.chat(userMessage);
            System.out.println("Assistant: " + assistantResponse);
        }
        scanner.close();
    }

    private static Path toPath(String resourceName) {
        try {
            URL resourceUrl = TestRoutage.class.getClassLoader().getResource(resourceName);
            if (resourceUrl == null) {
                throw new RuntimeException("Resource not found: " + resourceName);
            }
            return Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
