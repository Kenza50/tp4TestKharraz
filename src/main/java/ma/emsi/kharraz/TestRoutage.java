package ma.emsi.kharraz;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
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
        Document document = loadDocument(documentPath, new TextDocumentParser());

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddingModel.embedAll(segments).content(), segments);
        return embeddingStore;
    }

    public static void main(String[] args) {
        configureLogger();

        // Phase 1: Ingestion
        EmbeddingStore<TextSegment> iaStore = createEmbeddingStore("ia_content.txt");
        EmbeddingStore<TextSegment> cuisineStore = createEmbeddingStore("cuisine_content.txt");

        // Phase 2: Retrieval
        String llmKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ContentRetriever iaRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(iaStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever cuisineRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(cuisineStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        Map<ContentRetriever, String> retrieverMap = new HashMap<>();
        retrieverMap.put(iaRetriever, "Contient des informations sur l'intelligence artificielle, le machine learning et les modèles de langage.");
        retrieverMap.put(cuisineRetriever, "Contient des recettes de cuisine, des ingrédients et des instructions de préparation.");

        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverMap);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Create Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // Start chatting
        System.out.println("Assistant is ready. Ask your questions about AI or cooking.");
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
