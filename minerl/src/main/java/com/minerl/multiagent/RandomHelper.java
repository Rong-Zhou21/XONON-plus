package com.minerl.multiagent;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.LogManager;

import org.apache.commons.codec.binary.Hex;

public class RandomHelper {
    private static Random seedsRand = new Random(0);
    private static Map<String, Random> rands = new HashMap<>();
    private static int DEFAULT_LENGTH = 6;

    public static Random getRandom() {
        // TODO maybe instead generate a new Random with a seed from this rand for better thread safety or to prevent
        // accidental seeding elsewhere
        return getRandom("default");
    }

    public static Random getRandom(String key) {
        // TODO maybe instead generate a new Random with a seed from this rand for better thread safety or to prevent
        // accidental seeding elsewhere
        if (!rands.containsKey(key)) {
            rands.put(key, new Random(seedsRand.nextLong()));
        }
        return rands.get(key);
    }

    public static Random getRandom(String key, long defaultSeed) {
        // TODO maybe instead generate a new Random with a seed from this rand for better thread safety or to prevent
        // accidental seeding elsewhere
        if (!rands.containsKey(key)) {
            rands.put(key, new Random(defaultSeed));
        }
        return rands.get(key);
    }

    public static synchronized String getRandomHexString() {
        return getRandomHexString(DEFAULT_LENGTH);
    }

    public static synchronized String getRandomHexString(int nbytes) {
        byte[] buf = new byte[nbytes];
        getRandom().nextBytes(buf);
        return new String(Hex.encodeHexString(buf));
    }

    public static synchronized double randomDouble() {
        return getRandom().nextDouble();
    }


    public static long getSeed(Random rand) {
        try {
            Field seedField = Random.class.
                    getDeclaredField("seed");
            seedField.setAccessible(true);
            return ((AtomicLong) seedField.get(rand)).get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static double nextDouble() {
        return nextDouble("default");
    }

    public static double nextDouble(String key) {
        Random rand = getRandom(key);
        // System.out.println("nextDouble(" + key + "), seed = " + getSeed(rand));
        return getRandom(key).nextDouble();
    }

    public static float nextFloat() {
        return nextFloat("default");
    }

    public static float nextFloat(String key) {
        Random rand = getRandom(key);
        // System.out.println("nextDouble(" + key + "), seed = " + getSeed(rand));
        return getRandom(key).nextFloat();
    }

    public static long nextLong() {
        return nextLong("default");
    }

    public static long nextLong(String key) {
        Random rand = getRandom(key);
        // System.out.println("nextDouble(" + key + "), seed = " + getSeed(rand));
        return getRandom(key).nextLong();
    }
}
