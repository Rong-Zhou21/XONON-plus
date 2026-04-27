package net.minecraft.resources;

import net.minecraft.resources.data.IMetadataSectionSerializer;
import net.minecraft.util.ResourceLocation;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nullable;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class JavaResourcePack implements IResourcePack {
    private Set<String> resources;
    private Set<String> namespaces;
    private final String index = "index.txt";
    public JavaResourcePack() {
        try {
            resources = IOUtils.readLines(this.getClass().getClassLoader().getResourceAsStream(index)).stream().collect(Collectors.toSet());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        namespaces = resources.stream().map(s -> getResourceLocation(s).getNamespace()).collect(Collectors.toSet());
    }

    @Override
    public InputStream getRootResourceStream(String fileName) throws IOException {
        return null;
    }

    @Override
    public InputStream getResourceStream(ResourcePackType type, ResourceLocation location) throws IOException {
        return this.getClass().getClassLoader().getResourceAsStream(pathjoin(location.getNamespace(), location.getPath()));
    }

    @Override
    public Collection<ResourceLocation> getAllResourceLocations(ResourcePackType type, String namespaceIn, String pathIn, int maxDepthIn, Predicate<String> filterIn) {
        return resources.stream()
                .filter(s -> s.startsWith(pathjoin(namespaceIn, pathIn)))
                .filter(filterIn)
                .map(s -> getResourceLocation(s))
                .collect(Collectors.toList());
    }

    private static ResourceLocation getResourceLocation(String path) {
        String[] split = path.split("/");
        if (split.length < 2) {
            return new ResourceLocation(path);
        } else {
            String namespace = split[0];
            String resourcePath = path.substring(namespace.length() + 1);
            return new ResourceLocation(namespace, resourcePath);
        }
    }

    @Override
    public boolean resourceExists(ResourcePackType type, ResourceLocation location) {
        return resources.contains(pathjoin(location.getNamespace(), location.getPath()));
    }

    @Override
    public Set<String> getResourceNamespaces(ResourcePackType type) {
        return namespaces;
    }

    @Nullable
    @Override
    public <T> T getMetadata(IMetadataSectionSerializer<T> deserializer) throws IOException {
        return null;
    }

    @Override
    public String getName() {
        return "JavaPack";
    }

    @Override
    public void close() {

    }

    private static String pathjoin(String... args) {
        return String.join("/", args);
    }
}
